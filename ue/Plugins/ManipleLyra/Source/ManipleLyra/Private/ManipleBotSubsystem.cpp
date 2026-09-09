#include "ManipleBotSubsystem.h"
#include "ManipleLyra.h"
#include "ManipleLyraSchema.h"
#include "ManipleAgentComponent.h"
#include "ManipleTritonClient.h"
#include "Dom/JsonObject.h"
#include "Player/LyraPlayerBotController.h"
#include "Messages/LyraVerbMessage.h"
#include "Teams/LyraTeamSubsystem.h"
#include "Player/LyraPlayerState.h"
#include "HAL/PlatformMisc.h"
#include "AIController.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/PlayerState.h"
#include "GameFramework/GameStateBase.h"
#include "GameFramework/WorldSettings.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "NavigationSystem.h"
#include "Components/CapsuleComponent.h"
#include "Character/LyraCharacter.h"
#include "Physics/LyraCollisionChannels.h"
#include "HAL/PlatformTime.h"
#include "UObject/UObjectGlobals.h"

using namespace ManipleLyra;

// ---------- curriculum stages ----------

namespace
{
	// name, spawn, min, max, face, los, opponents, approach shaping, promote at kills/min
	const FManipleStage GStages[] = {
		{TEXT("duel"), FManipleStage::ESpawn::Pair, 500.f, 1000.f, true, true, EManipleOpponents::Self, true, 2.0f},
		{TEXT("close"), FManipleStage::ESpawn::Pair, 800.f, 2500.f, false, false, EManipleOpponents::Self, true, 1.5f},
		{TEXT("mid"), FManipleStage::ESpawn::Near, 0.f, 4000.f, false, false, EManipleOpponents::Self, false, 1.0f},
		{TEXT("full"), FManipleStage::ESpawn::Lyra, 0.f, 0.f, false, false, EManipleOpponents::Self, false, 0.f},
	};
	constexpr int32 NumStages = UE_ARRAY_COUNT(GStages);
	constexpr int32 KillWindowSlots = 60; // stats windows of 5 s game time = 5 game minutes
	constexpr int32 MinStageWindows = 360; // 30 game minutes in a stage before it can promote or demote
	constexpr float DemoteFraction = 0.25f; // demote when the kill rate drops below this share of the previous stage's threshold
	constexpr float ReplayShare = 0.3f; // share of placements that use a random earlier stage (keeps duel skills alive)
	constexpr int32 MaxPlaceAttempts = 20;

	// Lyra restarts the match by reloading the map (new world, new subsystem): keep the reached stage per process.
	int32 GPersistentStage = -1;
}

// ---------- lifecycle ----------

bool UManipleBotSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	if (!Super::ShouldCreateSubsystem(Outer))
		return false;
	const UWorld* World = Cast<UWorld>(Outer);
	if (!World || !(World->WorldType == EWorldType::Game || World->WorldType == EWorldType::PIE))
		return false;
	return FManipleBotConfig::FromCommandLine().IsActive();
}

// ---------- score span ----------

void FManipleScoreSpan::Add(const FManipleScoreSpan& O)
{
	GameSec += O.GameSec;
	AgentSec += O.AgentSec;
	Kills += O.Kills;
	Deaths += O.Deaths;
	Hits += O.Hits;
	Shots += O.Shots;
	Stat += O.Stat;
	Team += O.Team;
}

double FManipleScoreSpan::Score(EManipleScoreKind Kind) const
{
	const double Min = Minutes(Kind);
	if (Min <= 0.0)
		return 0.0;
	switch (Kind)
	{
	case EManipleScoreKind::Stat:
		return Stat / Min;
	case EManipleScoreKind::Team:
		return Team / Min;
	default:
		return Kills / Min;
	}
}

void UManipleBotSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	Config = FManipleBotConfig::FromCommandLine();
	Transitions = FManipleTransitionBatch(ObsDim, ActDim);
	if (Config.Brain == EManipleBrain::Triton)
	{
		Client = MakeShared<FManipleTritonClient>(Config.TritonUrl, 2.f);
		Agent = MakeShared<FManipleAgentClient>(Client, Config.Model);
	}
	UE_LOG(LogManipleLyra, Display, TEXT("bot subsystem active: %s"), *Config.ToString());
}

void UManipleBotSubsystem::Deinitialize()
{
	if (UWorld* World = GetWorld())
	{
		if (UGameplayMessageSubsystem* Msg =
				World->GetGameInstance() ? World->GetGameInstance()->GetSubsystem<UGameplayMessageSubsystem>() : nullptr)
		{
			Msg->UnregisterListener(DamageListener);
			Msg->UnregisterListener(EliminationListener);
		}
	}
	Agent.Reset();
	Client.Reset();
	Super::Deinitialize();
}

void UManipleBotSubsystem::OnWorldStarted()
{
	UWorld* World = GetWorld();

	// reward attribution + kill/death stats in every mode
	{
		UGameplayMessageSubsystem& Msg = UGameplayMessageSubsystem::Get(World);
		DamageListener =
			Msg.RegisterListener(FGameplayTag::RequestGameplayTag(TEXT("Lyra.Damage.Message")), this, &ThisClass::OnDamageMessage);
		EliminationListener = Msg.RegisterListener(
			FGameplayTag::RequestGameplayTag(TEXT("Lyra.Elimination.Message")), this, &ThisClass::OnEliminationMessage);
	}

	if (Config.ScoreKind != EManipleScoreKind::Kills)
	{
		ScoreTag = FGameplayTag::RequestGameplayTag(FName(*Config.ScoreTag), false);
		if (!ScoreTag.IsValid())
			UE_LOG(LogManipleLyra, Error, TEXT("-ManipleScore tag '%s' is not a registered gameplay tag: score stays 0"), *Config.ScoreTag);
	}

	if (Config.CurriculumStage >= 0)
	{
		const int32 Start =
			(Config.bCurriculumAuto && GPersistentStage > Config.CurriculumStage) ? GPersistentStage : Config.CurriculumStage;
		ApplyStage(FMath::Clamp(Start, 0, NumStages - 1));
	}

	if (Config.TimeScale != 1.f)
	{
		AWorldSettings* WS = World->GetWorldSettings();
		WS->MaxGlobalTimeDilation = FMath::Max(WS->MaxGlobalTimeDilation, Config.TimeScale);
		WS->SetTimeDilation(Config.TimeScale);
		UE_LOG(LogManipleLyra, Display, TEXT("time dilation set to %.2f"), WS->GetEffectiveTimeDilation());
	}
}

void UManipleBotSubsystem::Tick(float DeltaTime)
{
	UWorld* World = GetWorld();
	if (!World || !World->HasBegunPlay() || World->GetNetMode() == NM_Client)
		return;
	if (!bWorldStarted)
	{
		bWorldStarted = true;
		OnWorldStarted();
	}

	++WinFrames;
	WinFrameSec += DeltaTime;

	ScanTimer += DeltaTime;
	if (ScanTimer >= 0.5f)
	{
		ScanTimer = 0.f;
		ScanForBots();
	}

	if (Config.IsTraining() && !bRegistered && !bRegisterPending && FPlatformTime::Seconds() >= RegisterRetryAt)
		Register();

	DecisionTimer += DeltaTime;
	const float Period = 1.f / Config.DecisionHz;
	if (DecisionTimer >= Period)
	{
		DecisionTimer = FMath::Fmod(DecisionTimer, Period);
		Decide();
	}

	if (Config.bSpectate)
		UpdateSpectator(DeltaTime);

	StatsTimer += DeltaTime;
	if (StatsTimer >= 5.f)
	{
		StatsTimer = 0.f;
		LogStats();
	}

	UpdateEval(DeltaTime);
	if (QuitAt > 0.0 && FPlatformTime::Seconds() >= QuitAt)
	{
		QuitAt = 0.0;
		FPlatformMisc::RequestExit(false);
	}
}

// ---------- bots ----------

void UManipleBotSubsystem::SpawnExtraBots()
{
	// ULyraBotCreationComponent is not exported from LyraGame: reach it by reflection.
	AGameStateBase* GS = GetWorld()->GetGameState();
	UClass* BotClass = FindObject<UClass>(nullptr, TEXT("/Script/LyraGame.LyraBotCreationComponent"));
	UActorComponent* Comp = (GS && BotClass) ? GS->GetComponentByClass(BotClass) : nullptr;
	UFunction* Fn = Comp ? Comp->FindFunction(TEXT("SpawnOneBot")) : nullptr;
	if (!Fn)
	{
		UE_LOG(LogManipleLyra, Warning, TEXT("cannot spawn extra bots: LyraBotCreationComponent/SpawnOneBot not found"));
		return;
	}
	for (int32 i = 0; i < Config.SpawnBots; ++i)
		Comp->ProcessEvent(Fn, nullptr);
	UE_LOG(LogManipleLyra, Display, TEXT("spawned %d extra bots"), Config.SpawnBots);
}

void UManipleBotSubsystem::MakeEndless()
{
	// B_TeamDeathMatchScoring (game state component, Blueprint) ends the match on TargetScore / CountDownTime: push both out of reach.
	// Re-applied every scan: the Blueprint may reinitialise them when a phase starts.
	AGameStateBase* GS = GetWorld()->GetGameState();
	if (!GS)
		return;
	for (UActorComponent* Comp : GS->GetComponents())
	{
		if (!Comp || !Comp->GetClass()->GetName().Contains(TEXT("Scoring")))
			continue;
		for (const TCHAR* Name : {TEXT("TargetScore"), TEXT("CountDownTime")})
		{
			FProperty* P = Comp->GetClass()->FindPropertyByName(Name);
			void* Value = P ? P->ContainerPtrToValuePtr<void>(Comp) : nullptr;
			if (!Value)
				continue;
			double Old = 0.0;
			if (FIntProperty* IP = CastField<FIntProperty>(P))
			{
				Old = IP->GetPropertyValue(Value);
				if (Old < 1e8)
					IP->SetPropertyValue(Value, 1000000000);
			}
			else if (FFloatProperty* FP = CastField<FFloatProperty>(P))
			{
				Old = FP->GetPropertyValue(Value);
				if (Old < 1e8)
					FP->SetPropertyValue(Value, 1e9f);
			}
			else if (FDoubleProperty* DP = CastField<FDoubleProperty>(P))
			{
				Old = DP->GetPropertyValue(Value);
				if (Old < 1e8)
					DP->SetPropertyValue(Value, 1e9);
			}
			else
			{
				continue;
			}
			if (Old < 1e8)
			{
				UE_LOG(LogManipleLyra, Display, TEXT("endless match: %s.%s %.0f -> 1e9 (%s)%s"), *Comp->GetClass()->GetName(), Name, Old,
					*P->GetClass()->GetName(), bEndlessApplied ? TEXT(" [was reset]") : TEXT(""));
			}
			bEndlessApplied = true;
		}
	}
}

void UManipleBotSubsystem::ScanForBots()
{
	OwnedControllers.RemoveAll([](const TWeakObjectPtr<AAIController>& C) { return !C.IsValid(); });

	int32 BotControllers = 0;
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	for (TActorIterator<ALyraPlayerBotController> It(GetWorld()); It; ++It)
	{
		++BotControllers;
		AAIController* Ctrl = *It;
		APawn* Pawn = Ctrl->GetPawn();
		if (!Pawn || Pawn->FindComponentByClass<UManipleAgentComponent>())
			continue;
		if (Config.Opponents == EManipleOpponents::Lyra)
		{
			// the higher team keeps its behaviour tree; wait until the team is known
			const int32 Team = Teams ? Teams->FindTeamFromObject(Pawn) : INDEX_NONE;
			if (Team == INDEX_NONE || Team > 1)
				continue;
		}

		if (!OwnedControllers.Contains(Ctrl))
		{
			if (Config.MaxBots >= 0 && OwnedControllers.Num() >= Config.MaxBots)
				continue;
			OwnedControllers.Add(Ctrl);
		}
		UManipleAgentComponent* NewAgent = NewObject<UManipleAgentComponent>(Pawn, TEXT("ManipleAgent"));
		NewAgent->Init(NextAgentId++, this);
		AssignRole(*NewAgent);
		NewAgent->RegisterComponent();
	}

	// Lyra's own bots exist -> the experience is up; add ours once.
	if (!bSpawnedExtra && Config.SpawnBots > 0 && BotControllers > 0)
	{
		bSpawnedExtra = true;
		SpawnExtraBots();
	}
	if (Config.bEndless)
		MakeEndless();

	// keep dying agents around until their last transition is flushed; drop only dead objects
	Agents.RemoveAll([](const TWeakObjectPtr<UManipleAgentComponent>& A) { return !A.IsValid(); });
	for (const TWeakObjectPtr<AAIController>& C : OwnedControllers)
	{
		APawn* Pawn = C.IsValid() ? C->GetPawn() : nullptr;
		UManipleAgentComponent* A = Pawn ? Pawn->FindComponentByClass<UManipleAgentComponent>() : nullptr;
		if (A && !Agents.Contains(A))
			Agents.Add(A);
	}
}

// ---------- training protocol ----------

void UManipleBotSubsystem::Register()
{
	bRegisterPending = true;
	FManipleAgentSpec Spec;
	Spec.ObsDim = ObsDim;
	Spec.ActDim = ActDim;
	Spec.Preset = Config.NetPreset;
	Spec.Hidden = Config.Hidden;
	Spec.Activation = Config.Activation;
	Spec.bLayerNorm = Config.bLayerNorm;
	Spec.LogStdInit = Config.LogStdInit;
	Spec.Ppo.Add(TEXT("entropy_coef"), Config.EntropyCoef);
	Spec.ScoreSource = TEXT("report"); // 'best' = the game-mode score we report, not the shaped return

	TWeakObjectPtr<UManipleBotSubsystem> Self(this);
	Agent->Register(Spec,
		[Self](const FManipleAgentStatus& S)
		{
			if (!Self.IsValid())
				return;
			Self->bRegisterPending = false;
			Self->bRegistered = S.bOk;
			if (S.bOk)
			{
				UE_LOG(LogManipleLyra, Display, TEXT("registered policy '%s' v%lld: %s"), *Self->Config.Model, S.GetInt(TEXT("version")),
					*S.ToString());
			}
			else
			{
				UE_LOG(LogManipleLyra, Error, TEXT("register failed, retrying in 5 s: %s"), *S.Error);
				Self->RegisterRetryAt = FPlatformTime::Seconds() + 5.0;
			}
		});
}

void UManipleBotSubsystem::AddTransition(const UManipleAgentComponent& A, TConstArrayView<float> Obs, TConstArrayView<float> Action,
	float Reward, bool bDone, float LogP, int64 PolicyVersion)
{
	if (!Config.IsTraining() || A.bHeuristic)
		return;
	Transitions.Add(Obs, Action, Reward, bDone, A.GetAgentId(), A.GetEpisodeId(), PolicyVersion, LogP);
	WinRewardSum += Reward;
}

void UManipleBotSubsystem::OnEpisodeEnd(const UManipleAgentComponent& A)
{
	++WinEpisodes;
	WinReturnSum += A.GetEpisodeReturn();
}

void UManipleBotSubsystem::OnAgentDied(const UManipleAgentComponent& A)
{
	++WinDeaths;
}

void UManipleBotSubsystem::OnAgentShot(const UManipleAgentComponent& A)
{
	if (!A.bHeuristic)
		++WinShots;
}

void UManipleBotSubsystem::FlushTransitions()
{
	// every agent closes the transition it opened last tick; dead ones close their episode
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		UManipleAgentComponent* A = W.Get();
		if (!A || !A->HasTransition())
			continue;
		if (A->IsDead())
		{
			A->FlushTransition(true);
		}
		else if (A->IsReady())
		{
			const bool bTimeUp = A->GetEpisodeTime() > EpisodeTimeLimitSec;
			A->FlushTransition(bTimeUp);
			if (bTimeUp)
				A->bPlaced = false; // fresh start for the next episode
		}
	}
	if (Transitions.Num() == 0)
		return;

	WinRows += Transitions.Num();
	TWeakObjectPtr<UManipleBotSubsystem> Self(this);
	Agent->Observe(Transitions,
		[Self](const FManipleAgentStatus& S)
		{
			if (Self.IsValid() && S.bOk)
			{
				double Std = 0.0;
				const TSharedPtr<FJsonObject>* Stats = nullptr;
				if (S.Json->TryGetObjectField(TEXT("stats"), Stats) && Stats)
					(*Stats)->TryGetNumberField(TEXT("std"), Std);
				Self->LastTrainStatus = FString::Printf(TEXT("v%lld updates=%lld buffered=%lld dropped=%lld std=%.2f"),
					S.GetInt(TEXT("version")), S.GetInt(TEXT("updates")), S.GetInt(TEXT("buffered")), S.GetInt(TEXT("dropped_stale")), Std);
			}
		});
	Transitions.Reset();
}

// ---------- decision loop ----------

void UManipleBotSubsystem::Decide()
{
	if (Config.IsTraining() && !bRegistered)
		return;

	// curriculum placement for bots starting a life / episode
	const FManipleStage* StageDef = CurrentStage();
	if (StageDef && (StageDef->Spawn != FManipleStage::ESpawn::Lyra || Stage > 0))
	{
		for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		{
			if (W.IsValid() && W->IsReady() && !W->bPlaced)
				PlaceAgent(*W);
		}
	}

	AccumulateScoreTime(1.f / Config.DecisionHz);

	TArray<UManipleAgentComponent*> Ready;
	TArray<float> Obs;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		UManipleAgentComponent* A = W.Get();
		if (!A || !A->IsReady())
			continue;
		if (Config.Brain == EManipleBrain::Random)
		{
			A->SetRandomAction();
		}
		else if (Config.Brain == EManipleBrain::Heuristic || A->bHeuristic)
		{
			A->BuildObservation(Obs);
			A->SetHeuristicAction(Obs);
		}
		else
		{
			Ready.Add(A);
		}
	}
	if (Config.Brain == EManipleBrain::Random || Config.Brain == EManipleBrain::Heuristic)
	{
		WinTickLatencyMs.Add(0.0);
		++WinTicks;
		WinRequests += Agents.Num();
		return;
	}
	if (Ready.Num() == 0)
		return;

	// skip a tick if the previous one has not answered yet (server stalled): agents keep their last action
	if (ActInFlight > 0)
		return;

	// fresh observations first: the dense shaping is the outcome of the previous action and belongs to that transition
	TArray<float> Row;
	Obs.Reset();
	Obs.Reserve(Ready.Num() * ObsDim);
	TArray<TWeakObjectPtr<UManipleAgentComponent>> Targets;
	const bool bTrain = Config.IsTraining();
	for (UManipleAgentComponent* A : Ready)
	{
		A->BuildObservation(Row);
		Obs.Append(Row);
		Targets.Add(A);
		if (bTrain)
			A->AddShapingFromObservation();
	}
	if (bTrain)
		FlushTransitions();

	++ActInFlight;
	++WinRequests;
	TSharedRef<TArray<float>> ObsKeep = MakeShared<TArray<float>>(MoveTemp(Obs));
	TWeakObjectPtr<UManipleBotSubsystem> Self(this);
	auto OnAct = [Self, Targets, ObsKeep, bTrain](const FManipleActResult& R)
	{
		if (Self.IsValid())
		{
			--Self->ActInFlight;
			++Self->WinTicks;
			Self->WinTickLatencyMs.Add(R.LatencyMs);
			if (!R.bOk)
				++Self->WinFailures;
			else
				Self->LastPolicyVersion = R.PolicyVersion;
		}
		if (!R.bOk)
			return;
		for (int32 i = 0; i < Targets.Num(); ++i)
		{
			UManipleAgentComponent* A = Targets[i].Get();
			if (!A || !A->IsReady())
				continue;
			const TConstArrayView<float> Action = R.Row(i, ActDim);
			A->SetAction(Action);
			if (Self.IsValid())
			{
				++Self->WinActionRows;
				if (Action[ActFire] > 0.f)
					++Self->WinFireRows;
			}
			if (bTrain)
				A->BeginTransition(TConstArrayView<float>(*ObsKeep).Slice(i * ObsDim, ObsDim), Action,
					R.LogP.IsValidIndex(i) ? R.LogP[i] : 0.f, R.PolicyVersion);
		}
	};

	if (Config.bSyncAct)
	{
		// deterministic: the action is in place before this frame's component ticks, whatever the wall clock does
		OnAct(Agent->ActSync(*ObsKeep, Targets.Num(), ObsDim, Config.bExplore, Config.Channel));
		return;
	}
	Agent->Act(*ObsKeep, Targets.Num(), ObsDim, Config.bExplore, Config.Channel, OnAct);
}

// ---------- reward attribution ----------

UManipleAgentComponent* UManipleBotSubsystem::FindAgent(const UObject* Obj) const
{
	if (!Obj)
		return nullptr;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid() && W->Owns(Obj))
			return W.Get();
	}
	// weapons and projectiles: walk up to the pawn that owns them
	if (const AActor* Actor = Cast<AActor>(Obj))
	{
		if (Actor->GetInstigator() && Actor->GetInstigator() != Actor)
			return FindAgent(Actor->GetInstigator());
		if (Actor->GetOwner())
			return FindAgent(Actor->GetOwner());
	}
	return nullptr;
}

void UManipleBotSubsystem::OnDamageMessage(FGameplayTag Channel, const FLyraVerbMessage& Msg)
{
	UManipleAgentComponent* Dealer = FindAgent(Msg.Instigator);
	UManipleAgentComponent* Victim = FindAgent(Msg.Target);
	UE_LOG(LogManipleLyra, Verbose, TEXT("damage %.1f: %s (agent %d) -> %s (agent %d)"), Msg.Magnitude, *GetNameSafe(Msg.Instigator),
		Dealer ? Dealer->GetAgentId() : -1, *GetNameSafe(Msg.Target), Victim ? Victim->GetAgentId() : -1);
	if (Dealer == Victim)
		return; // self damage or none of ours
	const float Scaled = (float)Msg.Magnitude / 100.f; // Lyra characters have 100 max health
	if (Dealer)
	{
		++Dealer->Hits;
		if (!Dealer->bHeuristic)
			++WinHits;
		const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
		const bool bFriendly = Teams && Teams->CompareTeams(Msg.Instigator.Get(), Msg.Target.Get()) == ELyraTeamComparison::OnSameTeam;
		Dealer->AddReward((bFriendly ? -RewardDamageDealtScale : RewardDamageDealtScale) * Scaled);
	}
	if (Victim)
		Victim->AddReward(RewardDamageTakenScale * Scaled);
}

void UManipleBotSubsystem::OnEliminationMessage(FGameplayTag Channel, const FLyraVerbMessage& Msg)
{
	UManipleAgentComponent* Killer = FindAgent(Msg.Instigator);
	UE_LOG(LogManipleLyra, Verbose, TEXT("elimination: %s (agent %d) -> %s"), *GetNameSafe(Msg.Instigator),
		Killer ? Killer->GetAgentId() : -1, *GetNameSafe(Msg.Target));
	if (!Killer || Killer->Owns(Msg.Target))
		return; // death penalty comes from the agent's own health component
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	const bool bFriendly = Teams && Teams->CompareTeams(Msg.Instigator.Get(), Msg.Target.Get()) == ELyraTeamComparison::OnSameTeam;
	if (bFriendly)
	{
		Killer->AddReward(RewardTeamKill);
		return;
	}
	Killer->AddReward(RewardKill);
	++Killer->Kills;
	if (Killer->bHeuristic)
	{
		++WinHeuristicKills;
	}
	else
	{
		++WinKills;
		if (IsFinalPlacement(*Killer))
			++WinScore.Kills;
	}
}

// ---------- curriculum ----------

const FManipleStage* UManipleBotSubsystem::CurrentStage() const
{
	return Stage >= 0 && Stage < NumStages ? &GStages[Stage] : nullptr;
}

void UManipleBotSubsystem::ApplyStage(int32 NewStage)
{
	Stage = FMath::Clamp(NewStage, 0, NumStages - 1);
	GPersistentStage = Stage;
	KillHistory.Reset();
	StageAgeWindows = 0;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid())
		{
			AssignRole(*W);
			W->bPlaced = false;
		}
	}
	const FManipleStage& S = GStages[Stage];
	UE_LOG(LogManipleLyra, Display,
		TEXT("curriculum stage %d '%s': spawn=%d dist=%.0f-%.0f face=%d los=%d approach=%d promote_at=%.1f kills/min"), Stage, S.Name,
		(int32)S.Spawn, S.MinDist, S.MaxDist, S.bFaceEachOther ? 1 : 0, S.bRequireLineOfSight ? 1 : 0, S.bApproachShaping ? 1 : 0,
		S.PromoteKillsPerMin);
}

void UManipleBotSubsystem::AssignRole(UManipleAgentComponent& A) const
{
	const FManipleStage* S = CurrentStage();
	EManipleOpponents Opp = Config.Opponents;
	if (Opp == EManipleOpponents::Stage)
		Opp = S ? S->Opponents : EManipleOpponents::Self;

	bool bHeuristic = false;
	if (Opp == EManipleOpponents::Mixed)
	{
		bHeuristic = A.GetAgentId() % 2 == 1;
	}
	else if (Opp == EManipleOpponents::Heuristic)
	{
		// one whole team: the higher Lyra team id
		const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
		const int32 Team = Teams ? Teams->FindTeamFromObject(A.GetOwner()) : INDEX_NONE;
		bHeuristic = Team > 1;
	}
	A.bHeuristic = bHeuristic && Config.Brain == EManipleBrain::Triton;
	A.bApproachShaping = S && S->bApproachShaping;
}

const FManipleStage* UManipleBotSubsystem::PickPlacementStage() const
{
	if (Stage > 0 && FMath::FRand() < ReplayShare)
		return &GStages[FMath::RandHelper(Stage)];
	return CurrentStage();
}

void UManipleBotSubsystem::PlaceAgent(UManipleAgentComponent& A)
{
	const FManipleStage* S = PickPlacementStage();
	if (S)
		A.PlacedStage = (int32)(S - GStages);
	if (S && S->Spawn == FManipleStage::ESpawn::Lyra)
	{
		A.bPlaced = true; // replaying the 'full' stage means keeping Lyra's spawn
		return;
	}
	UWorld* World = GetWorld();
	const ULyraTeamSubsystem* Teams = World->GetSubsystem<ULyraTeamSubsystem>();
	UNavigationSystemV1* Nav = UNavigationSystemV1::GetCurrent(World);
	APawn* Me = Cast<APawn>(A.GetOwner());
	if (!S || !Nav || !Me || !Teams)
		return;

	int32& Attempts = PlaceAttempts.FindOrAdd(&A);
	if (++Attempts > MaxPlaceAttempts)
	{
		// give up for this episode (navmesh not ready, no enemy alive, ...); the next episode tries again
		A.bPlaced = true;
		PlaceAttempts.Remove(&A);
		UE_LOG(LogManipleLyra, Display, TEXT("agent %d: placement failed %d times, keeping Lyra's spawn this episode"), A.GetAgentId(),
			MaxPlaceAttempts);
		return;
	}

	// a random living enemy to start next to
	TArray<UManipleAgentComponent*> Enemies;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid() && W.Get() != &A && W->IsReady() && Teams->CompareTeams(Me, W->GetOwner()) == ELyraTeamComparison::DifferentTeams)
			Enemies.Add(W.Get());
	}
	if (Enemies.Num() == 0)
		return; // counts as an attempt: all of them dead is not going to fix itself this tick
	UManipleAgentComponent* E = Enemies[FMath::RandHelper(Enemies.Num())];
	const APawn* EnemyPawn = Cast<APawn>(E->GetOwner());
	const FVector EnemyLoc = EnemyPawn->GetActorLocation();

	const UCapsuleComponent* Capsule = Cast<ALyraCharacter>(Me) ? Cast<ALyraCharacter>(Me)->GetCapsuleComponent() : nullptr;
	const float HalfHeight = Capsule ? Capsule->GetScaledCapsuleHalfHeight() : 90.f;
	FCollisionQueryParams Params(SCENE_QUERY_STAT(ManiplePlacement), false, Me);

	for (int32 Try = 0; Try < 10; ++Try)
	{
		FNavLocation Out;
		if (!Nav->GetRandomReachablePointInRadius(EnemyLoc, S->MaxDist, Out))
			continue;
		const float Dist = FVector::Dist2D(Out.Location, EnemyLoc);
		if (Dist < S->MinDist)
			continue;
		const FVector Spot = Out.Location + FVector(0.f, 0.f, HalfHeight);
		if (S->bRequireLineOfSight)
		{
			FHitResult Hit;
			const FVector Eye = Spot + FVector(0.f, 0.f, 60.f);
			const bool bBlocked = World->LineTraceSingleByChannel(Hit, Eye, EnemyLoc, Lyra_TraceChannel_Weapon_Capsule, Params) &&
				Hit.GetActor() != EnemyPawn;
			if (bBlocked)
				continue;
		}
		A.Place(Spot, (EnemyLoc - Spot).Rotation().Yaw);
		++WinPlacements;
		if (S->bFaceEachOther)
			E->FaceTowards(Spot);
		PlaceAttempts.Remove(&A);
		UE_LOG(LogManipleLyra, Verbose, TEXT("agent %d placed %.0f cm from agent %d (stage %s)"), A.GetAgentId(), Dist, E->GetAgentId(),
			S->Name);
		return;
	}
}

void UManipleBotSubsystem::UpdateCurriculum()
{
	const FManipleStage* S = CurrentStage();
	if (!S)
		return;
	KillHistory.Add(WinKills);
	if (KillHistory.Num() > KillWindowSlots)
		KillHistory.RemoveAt(0, KillHistory.Num() - KillWindowSlots);

	int32 Sum = 0;
	for (int32 K : KillHistory)
		Sum += K;
	const double Minutes = KillHistory.Num() * 5.0 / 60.0;
	const double KillsPerMin = Minutes > 0 ? Sum / Minutes : 0.0;

	++StageAgeWindows;
	if (!Config.bCurriculumAuto || StageAgeWindows < MinStageWindows || KillHistory.Num() < KillWindowSlots)
		return;
	if (S->PromoteKillsPerMin > 0.f && KillsPerMin >= S->PromoteKillsPerMin)
	{
		UE_LOG(LogManipleLyra, Display, TEXT("curriculum: %.2f kills/min over %.0f min after %d min in stage, advancing"), KillsPerMin,
			Minutes, StageAgeWindows * 5 / 60);
		ApplyStage(Stage + 1);
	}
	else if (Stage > 0 && KillsPerMin < GStages[Stage - 1].PromoteKillsPerMin * DemoteFraction)
	{
		UE_LOG(LogManipleLyra, Display, TEXT("curriculum: %.2f kills/min over %.0f min after %d min in stage, going back"), KillsPerMin,
			Minutes, StageAgeWindows * 5 / 60);
		ApplyStage(Stage - 1);
	}
}

// ---------- spectator ----------

void UManipleBotSubsystem::UpdateSpectator(float DeltaTime)
{
	SpectateTimer += DeltaTime;
	if (SpectateTimer < 8.f)
		return;
	SpectateTimer = 0.f;

	APlayerController* PC = GetWorld()->GetFirstPlayerController();
	if (!PC || PC->GetPawn())
		return; // a playing human keeps their own camera
	TArray<APawn*> Pawns;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid() && W->IsReady())
			Pawns.Add(Cast<APawn>(W->GetOwner()));
	}
	if (Pawns.Num() == 0)
		return;
	SpectateIndex = (SpectateIndex + 1) % Pawns.Num();
	PC->SetViewTargetWithBlend(Pawns[SpectateIndex], 0.5f);
}

// ---------- stats ----------

void UManipleBotSubsystem::LogStats()
{
	int32 Alive = 0;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		if (W.IsValid() && W->IsReady())
			++Alive;

	double Avg = 0, P95 = 0, Max = 0;
	if (WinTickLatencyMs.Num() > 0)
	{
		WinTickLatencyMs.Sort();
		for (double L : WinTickLatencyMs)
			Avg += L;
		Avg /= WinTickLatencyMs.Num();
		P95 = WinTickLatencyMs[FMath::Min(WinTickLatencyMs.Num() - 1, (int32)(WinTickLatencyMs.Num() * 0.95))];
		Max = WinTickLatencyMs.Last();
	}
	const TCHAR* Mode = Config.Brain == EManipleBrain::Random ? TEXT("random")
		: Config.Brain == EManipleBrain::Heuristic			  ? TEXT("heuristic")
		: Config.IsTraining()								  ? TEXT("train")
															  : TEXT("infer");
	const double Sec = FMath::Max(WinFrameSec, 1e-6);
	UE_LOG(LogManipleLyra, Display,
		TEXT("bench: mode=%s agents=%d alive=%d hz=%.0f ticks=%d lat_avg=%.2f lat_p95=%.2f lat_max=%.2f frame_avg=%.2f req_s=%.1f "
			 "failures=%d version=%lld"),
		Mode, Agents.Num(), Alive, Config.DecisionHz, WinTicks, Avg, P95, Max, WinFrames > 0 ? WinFrameSec / WinFrames * 1000.0 : 0.0,
		WinRequests / Sec, WinFailures, LastPolicyVersion);
	int32 KillSum = 0;
	for (int32 K : KillHistory)
		KillSum += K;
	const double KillsPerMin = KillHistory.Num() > 0 ? KillSum / (KillHistory.Num() * 5.0 / 60.0) : 0.0;
	int32 Dry = 0;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		if (W.IsValid() && W->IsReady() && W->IsDry())
			++Dry;
	SampleScore();
	WinScore.Deaths = WinDeaths;
	WinScore.Hits = WinHits;
	WinScore.Shots = WinShots;
	const FManipleStage* S = CurrentStage();
	UE_LOG(LogManipleLyra, Display,
		TEXT("train: stage=%s rows=%d reward_sum=%.3f fire=%.2f shots=%d hits=%d episodes=%d return_avg=%.3f placed=%d kills=%d "
			 "heuristic_kills=%d deaths=%d kills_per_min=%.2f score=%.3f score_min=%.2f dry=%d trainer=[%s]"),
		S ? S->Name : TEXT("off"), WinRows, WinRewardSum, WinActionRows > 0 ? (double)WinFireRows / WinActionRows : 0.0, WinShots, WinHits,
		WinEpisodes, WinEpisodes > 0 ? WinReturnSum / WinEpisodes : 0.0, WinPlacements, WinKills, WinHeuristicKills, WinDeaths, KillsPerMin,
		WinScore.Score(Config.ScoreKind), WinScore.Minutes(Config.ScoreKind), Dry, *LastTrainStatus);
	UpdateCurriculum();

	ReportScore.Add(WinScore);
	EvalScore.Add(WinScore);
	WinScore.Reset();
	if (++ReportWindows >= KillWindowSlots)
	{
		ReportWindows = 0;
		SendReport();
	}

	WinTicks = WinRequests = WinFailures = WinFrames = WinRows = WinEpisodes = WinKills = WinHeuristicKills = WinDeaths = 0;
	WinActionRows = WinFireRows = WinPlacements = WinShots = WinHits = 0;
	WinFrameSec = WinReturnSum = WinRewardSum = 0.0;
	WinTickLatencyMs.Reset();
}

// ---------- score ----------

bool UManipleBotSubsystem::IsFinalPlacement(const UManipleAgentComponent& A) const
{
	return A.PlacedStage < 0 || A.PlacedStage == NumStages - 1;
}

int32 UManipleBotSubsystem::PolicyTeamId() const
{
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	if (!Teams)
		return INDEX_NONE;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		if (W.IsValid() && !W->bHeuristic)
			return Teams->FindTeamFromObject(W->GetOwner());
	}
	return INDEX_NONE;
}

void UManipleBotSubsystem::AccumulateScoreTime(float Seconds)
{
	bool bAnyDamageable = false;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
	{
		const UManipleAgentComponent* A = W.Get();
		if (!A || A->bHeuristic || !A->IsReady() || A->HasDamageImmunity())
			continue;
		bAnyDamageable = true;
		if (IsFinalPlacement(*A))
			WinScore.AgentSec += Seconds;
	}
	if (bAnyDamageable)
		WinScore.GameSec += Seconds;
}

void UManipleBotSubsystem::SampleScore()
{
	if (Config.ScoreKind == EManipleScoreKind::Kills || !ScoreTag.IsValid())
		return;

	if (Config.ScoreKind == EManipleScoreKind::Stat)
	{
		for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		{
			const UManipleAgentComponent* A = W.Get();
			if (!A || A->bHeuristic || !IsFinalPlacement(*A))
				continue;
			ALyraPlayerState* PS = Cast<ALyraPlayerState>(A->GetPlayerState());
			if (!PS)
				continue;
			const int32 Count = PS->GetStatTagStackCount(ScoreTag);
			if (const int32* Last = LastStatCount.Find(PS))
				WinScore.Stat += Count - *Last;
			LastStatCount.Add(PS, Count);
		}
		return;
	}

	// Team: whole-team tag, only meaningful when the policy team plays the real game (final stage or no curriculum)
	if (Stage >= 0 && Stage != NumStages - 1)
		return;
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	const int32 Ours = PolicyTeamId();
	if (!Teams || Ours == INDEX_NONE)
		return;
	int32 BestOther = 0;
	for (int32 Team : Teams->GetTeamIDs())
	{
		if (Team != Ours)
			BestOther = FMath::Max(BestOther, Teams->GetTeamTagStackCount(Team, ScoreTag));
	}
	const double Diff = Teams->GetTeamTagStackCount(Ours, ScoreTag) - BestOther;
	if (bTeamDiffValid)
		WinScore.Team += Diff - LastTeamDiff;
	LastTeamDiff = Diff;
	bTeamDiffValid = true;
}

void UManipleBotSubsystem::SendReport()
{
	const FManipleScoreSpan Span = ReportScore;
	ReportScore.Reset();
	if (!Config.IsTraining() || !bRegistered || !Agent.IsValid() || LastPolicyVersion <= 0)
		return;
	const double Minutes = Span.Minutes(Config.ScoreKind);
	if (Minutes < 0.5)
		return; // nobody played the real game in this span (earlier curriculum stages only)

	const double Score = Span.Score(Config.ScoreKind);
	const int64 Weight = FMath::Max<int64>(1, FMath::RoundToInt64(Minutes));
	UE_LOG(LogManipleLyra, Display, TEXT("report: version=%lld score=%.3f (%s) minutes=%.1f kills=%d"), LastPolicyVersion, Score,
		*Config.ScoreToString(), Minutes, Span.Kills);
	Agent->Report(LastPolicyVersion, (float)Score, Weight,
		[](const FManipleAgentStatus& S)
		{
			if (!S.bOk)
				UE_LOG(LogManipleLyra, Warning, TEXT("report failed: %s"), *S.Error);
		});
}

// ---------- evaluation ----------

void UManipleBotSubsystem::UpdateEval(float DeltaTime)
{
	if (Config.EvalSeconds <= 0.f || bEvalDone)
		return;

	if (!bEvalStarted)
	{
		for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		{
			const UManipleAgentComponent* A = W.Get();
			if (A && !A->bHeuristic && A->IsReady() && !A->HasDamageImmunity())
			{
				bEvalStarted = true;
				break;
			}
		}
		if (bEvalStarted)
		{
			EvalScore.Reset();
			UE_LOG(LogManipleLyra, Display, TEXT("eval: started, %.0f game-seconds"), Config.EvalSeconds);
		}
		return;
	}

	EvalElapsed += DeltaTime;
	if (EvalElapsed >= Config.EvalSeconds)
		FinishEval();
}

void UManipleBotSubsystem::FinishEval()
{
	bEvalDone = true;
	LogStats(); // folds the partial window into EvalScore

	int32 PolicyAgents = 0;
	for (const TWeakObjectPtr<UManipleAgentComponent>& W : Agents)
		if (W.IsValid() && !W->bHeuristic)
			++PolicyAgents;

	const TCHAR* Mode = Config.Brain == EManipleBrain::Random ? TEXT("random")
		: Config.Brain == EManipleBrain::Heuristic			  ? TEXT("heuristic")
		: Config.IsTraining()								  ? TEXT("train")
															  : TEXT("infer");
	const FManipleStage* S = CurrentStage();
	const double Score = EvalScore.Score(Config.ScoreKind);
	const double Minutes = EvalScore.Minutes(Config.ScoreKind);
	UE_LOG(LogManipleLyra, Display,
		TEXT("eval: mode=%s model=%s version=%lld channel=%s opponents=%s stage=%s score=%s value=%.3f agents=%d game_min=%.2f "
			 "agent_min=%.2f kills=%d deaths=%d hits=%d shots=%d"),
		Mode, *Config.Model, LastPolicyVersion, Config.IsTraining() ? TEXT("latest") : *Config.Channel, Config.OpponentsToString(),
		S ? S->Name : TEXT("off"), *Config.ScoreToString(), Score, PolicyAgents, EvalScore.GameSec / 60.0, EvalScore.AgentSec / 60.0,
		EvalScore.Kills, EvalScore.Deaths, EvalScore.Hits, EvalScore.Shots);

	if (Config.bEvalReport && Agent.IsValid() && LastPolicyVersion > 0 && Minutes > 0.0)
	{
		const int64 Weight = FMath::Max<int64>(1, FMath::RoundToInt64(Minutes));
		TWeakObjectPtr<UManipleBotSubsystem> Self(this);
		Agent->Report(LastPolicyVersion, (float)Score, Weight,
			[Self](const FManipleAgentStatus& St)
			{
				if (St.bOk)
				{
					UE_LOG(LogManipleLyra, Display, TEXT("eval: score reported: %s"), *St.ToString());
				}
				else
				{
					UE_LOG(LogManipleLyra, Warning, TEXT("eval: report failed: %s"), *St.Error);
				}
				if (Self.IsValid())
					Self->RequestQuit(TEXT("evaluation reported"));
			});
		QuitAt = FPlatformTime::Seconds() + 10.0; // in case the reply never comes
		return;
	}
	RequestQuit(TEXT("evaluation done"));
}

void UManipleBotSubsystem::RequestQuit(const TCHAR* Why)
{
	UE_LOG(LogManipleLyra, Display, TEXT("quitting: %s"), Why);
	QuitAt = FPlatformTime::Seconds(); // next tick, outside any callback
}
