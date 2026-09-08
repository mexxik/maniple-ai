#include "ManipleAgentComponent.h"
#include "ManipleLyra.h"
#include "ManipleBotSubsystem.h"
#include "Character/LyraCharacter.h"
#include "Character/LyraHealthComponent.h"
#include "Teams/LyraTeamSubsystem.h"
#include "Physics/LyraCollisionChannels.h"
#include "AbilitySystem/LyraAbilitySystemComponent.h"
#include "AIController.h"
#include "BrainComponent.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "GameFramework/PlayerState.h"
#include "CollisionQueryParams.h"
#include "GameplayTagContainer.h"
#include "GameplayAbilitySpec.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "UObject/UnrealType.h"

using namespace ManipleLyra;

UManipleAgentComponent::UManipleAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UManipleAgentComponent::Init(int32 InAgentId, UManipleBotSubsystem* InSubsystem)
{
	AgentId = InAgentId;
	Subsystem = InSubsystem;
}

// ---------- lifecycle ----------

void UManipleAgentComponent::BeginPlay()
{
	Super::BeginPlay();
	Character = Cast<ALyraCharacter>(GetOwner());
	AI = Character.IsValid() ? Cast<AAIController>(Character->GetController()) : nullptr;
	if (!Character.IsValid() || !AI.IsValid())
	{
		UE_LOG(LogManipleLyra, Warning, TEXT("agent %d: owner is not an AI-controlled LyraCharacter, disabling"), AgentId);
		SetComponentTickEnabled(false);
		return;
	}
	TakeOverFromBehaviorTree();
	Aim = FRotator(0.f, AI->GetControlRotation().Yaw, 0.f);
	if (ULyraHealthComponent* H = ULyraHealthComponent::FindHealthComponent(Character.Get()))
		H->OnDeathStarted.AddDynamic(this, &UManipleAgentComponent::OnDeathStarted);
	UE_LOG(LogManipleLyra, Verbose, TEXT("agent %d: took over %s (%s)"), AgentId, *Character->GetName(), *AI->GetName());
}

void UManipleAgentComponent::EndPlay(const EEndPlayReason::Type Reason)
{
	// the pawn is going away: whatever is pending ends the episode
	if (bHasTransition)
		FlushTransition(true);
	Super::EndPlay(Reason);
}

void UManipleAgentComponent::OnDeathStarted(AActor* OwningActor)
{
	if (bDead)
		return;
	bDead = true;
	++Deaths;
	UE_LOG(LogManipleLyra, Verbose, TEXT("agent %d died"), AgentId);
	if (Subsystem.IsValid())
		Subsystem->OnAgentDied(*this);
	AddReward(RewardDeath);
	bHasAction = false;
	if (bFiring)
	{
		if (ULyraAbilitySystemComponent* ASC = Character.IsValid() ? Character->GetLyraAbilitySystemComponent() : nullptr)
		{
			static const FGameplayTag FireTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Fire"));
			ASC->AbilityInputTagReleased(FireTag);
		}
		bFiring = false;
	}
}

void UManipleAgentComponent::TakeOverFromBehaviorTree()
{
	if (UBrainComponent* Brain = AI->FindComponentByClass<UBrainComponent>())
	{
		if (Brain->IsRunning())
			Brain->StopLogic(TEXT("Maniple"));
	}
	AI->StopMovement();
	AI->ClearFocus(EAIFocusPriority::Gameplay);
	// AAIController::UpdateControlRotation would otherwise overwrite our aim with the pawn's yaw (and pitch 0) every tick
	AI->bSetControlRotationFromPawnOrientation = false;
}

bool UManipleAgentComponent::IsReady() const
{
	if (bDead || !Character.IsValid() || !AI.IsValid())
		return false;
	const ULyraHealthComponent* H = ULyraHealthComponent::FindHealthComponent(Character.Get());
	return H && !H->IsDeadOrDying();
}

// ---------- identity ----------

APlayerState* UManipleAgentComponent::GetPlayerState() const
{
	if (AI.IsValid() && AI->PlayerState)
		return AI->PlayerState;
	return Character.IsValid() ? Character->GetPlayerState() : nullptr;
}

bool UManipleAgentComponent::Owns(const UObject* Obj) const
{
	if (!Obj)
		return false;
	return Obj == Character.Get() || Obj == AI.Get() || Obj == GetPlayerState();
}

// ---------- observation ----------

void UManipleAgentComponent::BuildObservation(TArray<float>& Obs) const
{
	Obs.SetNumZeroed(ObsDim);
	const ALyraCharacter* Me = Character.Get();
	const FRotator Frame(0.f, Aim.Yaw, 0.f);
	const FVector MyLoc = Me->GetActorLocation();

	const FVector LocalVel = Frame.UnrotateVector(Me->GetVelocity()) / VelScale;
	Obs[ObsVel + 0] = LocalVel.X;
	Obs[ObsVel + 1] = LocalVel.Y;
	Obs[ObsVel + 2] = LocalVel.Z;

	if (const ULyraHealthComponent* H = ULyraHealthComponent::FindHealthComponent(Me))
	{
		Obs[ObsHealth] = H->GetMaxHealth() > 0.f ? H->GetHealth() / H->GetMaxHealth() : 0.f;
	}

	struct FEnemy
	{
		const ALyraCharacter* C;
		float Dist;
		float Health;
	};
	TArray<FEnemy> Enemies;
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	for (TActorIterator<ALyraCharacter> It(GetWorld()); It; ++It)
	{
		const ALyraCharacter* Other = *It;
		if (Other == Me)
			continue;
		const ULyraHealthComponent* OH = ULyraHealthComponent::FindHealthComponent(Other);
		if (!OH || OH->IsDeadOrDying())
			continue;
		if (Teams && Teams->CompareTeams(Me, Other) != ELyraTeamComparison::DifferentTeams)
			continue;
		Enemies.Add({Other, (float)FVector::Dist(MyLoc, Other->GetActorLocation()),
			OH->GetMaxHealth() > 0.f ? OH->GetHealth() / OH->GetMaxHealth() : 0.f});
	}
	Enemies.Sort([](const FEnemy& A, const FEnemy& B) { return A.Dist < B.Dist; });
	LastNearestDist = Enemies.Num() > 0 ? Enemies[0].Dist : -1.f;

	FCollisionQueryParams Params(SCENE_QUERY_STAT(ManipleVisibility), false, Me);
	const FVector Eye = Me->GetPawnViewLocation();
	const FVector View = Aim.Vector();
	LastAim = 0.f;
	for (int32 i = 0; i < FMath::Min(NumEnemies, Enemies.Num()); ++i)
	{
		const FEnemy& E = Enemies[i];
		const FVector Target = E.C->GetActorLocation();
		const FVector Rel = Frame.UnrotateVector(Target - MyLoc) / PosScale;
		FHitResult Hit;
		// weapon capsule channel: pawn capsules and world geometry block it, so "visible" means "can be shot"
		const bool bBlocked =
			GetWorld()->LineTraceSingleByChannel(Hit, Eye, Target, Lyra_TraceChannel_Weapon_Capsule, Params) && Hit.GetActor() != E.C;
		float* O = &Obs[ObsEnemies + i * EnemyStride];
		O[0] = Rel.X;
		O[1] = Rel.Y;
		O[2] = Rel.Z;
		O[3] = E.Dist / PosScale;
		O[4] = bBlocked ? 0.f : 1.f;
		O[5] = E.Health;
		if (!bBlocked)
			LastAim = FMath::Max(LastAim, (float)FVector::DotProduct(View, (Target - Eye).GetSafeNormal()));
		if (i == 0)
		{
			const FRotator Want = (Target - Eye).Rotation();
			LastYawErr = FRotator::NormalizeAxis(Want.Yaw - Aim.Yaw);
			LastPitchErr = FRotator::NormalizeAxis(Want.Pitch - Aim.Pitch);
			bLastVisible = !bBlocked;
		}
	}
	Obs[ObsPitch] = FRotator::NormalizeAxis(Aim.Pitch) / 90.f;
	Obs[ObsBias] = 1.f;

	// wall sensors: how far can I go in each direction (pawns are ignored, only geometry counts)
	FCollisionQueryParams RayParams(SCENE_QUERY_STAT(ManipleRays), false, Me);
	FCollisionObjectQueryParams RayObjects(ECC_WorldStatic);
	RayObjects.AddObjectTypesToQuery(ECC_WorldDynamic);
	for (int32 i = 0; i < NumRays; ++i)
	{
		const FVector Dir = Frame.RotateVector(FRotator(0.f, i * (360.f / NumRays), 0.f).Vector());
		FHitResult Hit;
		const bool bHit = GetWorld()->LineTraceSingleByObjectType(Hit, Eye, Eye + Dir * RayRange, RayObjects, RayParams);
		Obs[ObsRays + i] = bHit ? Hit.Distance / RayRange : 1.f;
	}
}

// ---------- actions ----------

void UManipleAgentComponent::SetAction(TConstArrayView<float> InAction)
{
	if (InAction.Num() < ActDim)
		return;
	for (int32 i = 0; i < ActDim; ++i)
		Action[i] = InAction[i];
	bHasAction = true;
}

void UManipleAgentComponent::SetRandomAction()
{
	Action[ActMoveFwd] = FMath::FRandRange(-1.f, 1.f);
	Action[ActMoveRight] = FMath::FRandRange(-1.f, 1.f);
	Action[ActYawRate] = FMath::FRandRange(-1.f, 1.f);
	Action[ActPitchRate] = FMath::FRandRange(-0.3f, 0.3f);
	Action[ActFire] = FMath::FRand() < 0.3f ? 1.f : -1.f;
	bHasAction = true;
}

// ---------- weapon (reflection into LyraQuickBarComponent / LyraInventoryItemInstance) ----------

UObject* UManipleAgentComponent::GetActiveWeaponItem() const
{
	static UClass* QuickBarClass = FindObject<UClass>(nullptr, TEXT("/Script/LyraGame.LyraQuickBarComponent"));
	UActorComponent* QuickBar = (AI.IsValid() && QuickBarClass) ? AI->GetComponentByClass(QuickBarClass) : nullptr;
	UFunction* Fn = QuickBar ? QuickBar->FindFunction(TEXT("GetActiveSlotItem")) : nullptr;
	if (!Fn)
		return nullptr;
	struct
	{
		UObject* ReturnValue = nullptr;
	} Params;
	QuickBar->ProcessEvent(Fn, &Params);
	return Params.ReturnValue;
}

int32 UManipleAgentComponent::GetStat(UObject* Item, const TCHAR* Tag) const
{
	UFunction* Fn = Item ? Item->FindFunction(TEXT("GetStatTagStackCount")) : nullptr;
	if (!Fn)
		return -1;
	struct
	{
		FGameplayTag Tag;
		int32 ReturnValue = 0;
	} Params;
	Params.Tag = FGameplayTag::RequestGameplayTag(Tag);
	Item->ProcessEvent(Fn, &Params);
	return Params.ReturnValue;
}

int32 UManipleAgentComponent::GetMagazineAmmo(UObject* Item) const
{
	return GetStat(Item, TEXT("Lyra.ShooterGame.Weapon.MagazineAmmo"));
}

void UManipleAgentComponent::RefillAmmo()
{
	// a pawn lives through many episodes (60 s truncation) and a policy that sprays empties 15 + 45 rounds in seconds:
	// top the spare ammo up at every episode start so ammo is a per-episode budget, not a one-off
	UObject* Item = GetActiveWeaponItem();
	UFunction* Fn = Item ? Item->FindFunction(TEXT("AddStatTagStack")) : nullptr;
	if (!Fn)
		return;
	const int32 Spare = GetStat(Item, TEXT("Lyra.ShooterGame.Weapon.SpareAmmo"));
	if (Spare < 0 || Spare >= SpareAmmoPerEpisode)
		return;
	struct
	{
		FGameplayTag Tag;
		int32 StackCount = 0;
	} Params;
	Params.Tag = FGameplayTag::RequestGameplayTag(TEXT("Lyra.ShooterGame.Weapon.SpareAmmo"));
	Params.StackCount = SpareAmmoPerEpisode - Spare;
	Item->ProcessEvent(Fn, &Params);
}

void UManipleAgentComponent::UpdateWeapon(float DeltaTime)
{
	// Lyra's bots reload through their behaviour tree; we do the same when the magazine runs dry
	ULyraAbilitySystemComponent* ASC = Character->GetLyraAbilitySystemComponent();
	if (!ASC)
		return;
	static const FGameplayTag ReloadTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Reload"));
	if (bReloading)
	{
		ASC->AbilityInputTagReleased(ReloadTag);
		bReloading = false;
	}
	UObject* Item = GetActiveWeaponItem();
	const int32 Magazine = GetMagazineAmmo(Item);
	if (Magazine >= 0 && LastMagazine > Magazine)
	{
		// a shot actually left the gun this frame: that is what the on-target bonus is for
		++Shots;
		AddReward(RewardShotCost);
		if (Subsystem.IsValid())
			Subsystem->OnAgentShot(*this);
		if (bLastVisible && FMath::Abs(LastYawErr) < OnTargetDeg && FMath::Abs(LastPitchErr) < OnTargetDeg)
			AddReward(Character->GetVelocity().Size() < 50.f ? RewardShotOnTarget : 0.5f * RewardShotOnTarget);
	}
	LastMagazine = Magazine;

	// reload once when the magazine runs dry, retry every 3 s while still empty; fire stays off meanwhile
	if (Item && Magazine == 0)
	{
		ReloadRetryTimer -= DeltaTime;
		if (!bMagazineEmpty || ReloadRetryTimer <= 0.f)
		{
			ASC->AbilityInputTagPressed(ReloadTag);
			bReloading = true;
			ReloadRetryTimer = 3.f;
		}
		bMagazineEmpty = true;
	}
	else
	{
		bMagazineEmpty = false;
	}

	DebugTimer += DeltaTime;
	if (DebugTimer > 3.f)
	{
		DebugTimer = 0.f;
		LogWeaponState();
		const UCharacterMovementComponent* Move = Character->GetCharacterMovement();
		FGameplayTagContainer Tags;
		if (const ULyraAbilitySystemComponent* A = Character->GetLyraAbilitySystemComponent())
			A->GetOwnedGameplayTags(Tags);
		UE_LOG(LogManipleLyra, VeryVerbose, TEXT("agent %d move: loc=%s vel=%.0f mode=%d maxspeed=%.0f input=%s act=(%.2f, %.2f) tags=%s"),
			AgentId, *Character->GetActorLocation().ToCompactString(), Character->GetVelocity().Size(),
			Move ? (int32)Move->MovementMode : -1, Move ? Move->GetMaxSpeed() : 0.f,
			*Character->GetPendingMovementInputVector().ToCompactString(), Action[ActMoveFwd], Action[ActMoveRight],
			*Tags.ToStringSimple());
	}
}

void UManipleAgentComponent::LogWeaponState() const
{
	const ULyraAbilitySystemComponent* ASC = Character->GetLyraAbilitySystemComponent();
	static const FGameplayTag FireTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Fire"));
	static const FGameplayTag ReloadTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Reload"));
	int32 Total = 0, Fire = 0, FireActive = 0, Reload = 0, ReloadActive = 0;
	if (ASC)
	{
		for (const FGameplayAbilitySpec& Spec : ASC->GetActivatableAbilities())
		{
			++Total;
			if (Spec.GetDynamicSpecSourceTags().HasTagExact(FireTag))
			{
				++Fire;
				FireActive += Spec.IsActive() ? 1 : 0;
			}
			if (Spec.GetDynamicSpecSourceTags().HasTagExact(ReloadTag))
			{
				++Reload;
				ReloadActive += Spec.IsActive() ? 1 : 0;
			}
		}
	}
	UObject* Item = GetActiveWeaponItem();
	UE_LOG(LogManipleLyra, Verbose,
		TEXT("agent %d weapon: abilities=%d fire=%d/%d reload=%d/%d item=%s magazine=%d spare=%d empty=%d shots=%d hits=%d firing=%d"),
		AgentId, Total, FireActive, Fire, ReloadActive, Reload, *GetNameSafe(Item), GetMagazineAmmo(Item),
		GetStat(Item, TEXT("Lyra.ShooterGame.Weapon.SpareAmmo")), bMagazineEmpty ? 1 : 0, Shots, Hits, bFiring ? 1 : 0);
}

void UManipleAgentComponent::SetHeuristicAction(TConstArrayView<float> Obs)
{
	// nearest enemy is slot 0; rel x/y/z are in the agent's yaw frame, scaled by PosScale
	const float* E = &Obs[ObsEnemies];
	const bool bHasEnemy = E[3] > 0.f;
	const bool bVisible = bHasEnemy && E[4] > 0.5f;

	Action[ActMoveFwd] = 1.f;
	Action[ActMoveRight] = 0.f;
	Action[ActFire] = -1.f;
	Action[ActPitchRate] = FMath::Clamp(-Obs[ObsPitch] * 90.f / 30.f, -1.f, 1.f); // level the view
	Action[ActYawRate] = 0.f;

	float YawErrDeg = 0.f, PitchErrDeg = 0.f;
	if (bVisible)
	{
		// track and shoot
		YawErrDeg = FMath::RadiansToDegrees(FMath::Atan2(E[1], E[0]));
		const float Flat = FMath::Sqrt(E[0] * E[0] + E[1] * E[1]);
		PitchErrDeg = FMath::RadiansToDegrees(FMath::Atan2(E[2], FMath::Max(Flat, 1e-4f))) - Obs[ObsPitch] * 90.f;
		Action[ActYawRate] = FMath::Clamp(YawErrDeg / 30.f, -1.f, 1.f);
		Action[ActPitchRate] = FMath::Clamp(PitchErrDeg / 30.f, -1.f, 1.f);
		Action[ActMoveFwd] = E[3] * PosScale > 1500.f ? 1.f : 0.f;
		Action[ActFire] = (FMath::Abs(YawErrDeg) < 5.f && FMath::Abs(PitchErrDeg) < 5.f) ? 1.f : -1.f;
	}
	else
	{
		// wander towards the enemy but follow open space: prefer the ray closest to the enemy direction that is not blocked
		const float WantDeg = bHasEnemy ? FMath::RadiansToDegrees(FMath::Atan2(E[1], E[0])) : 0.f;
		float BestScore = -1e9f;
		float BestDeg = 0.f;
		for (int32 i = 0; i < NumRays; ++i)
		{
			const float Deg = FRotator::NormalizeAxis(i * (360.f / NumRays));
			const float Open = Obs[ObsRays + i]; // 0..1
			const float Score = Open * 2.f - FMath::Abs(FRotator::NormalizeAxis(Deg - WantDeg)) / 180.f;
			if (Score > BestScore)
			{
				BestScore = Score;
				BestDeg = Deg;
			}
		}
		YawErrDeg = BestDeg;
		Action[ActYawRate] = FMath::Clamp(BestDeg / 45.f, -1.f, 1.f);
		if (Obs[ObsRays] < 0.15f) // wall right ahead: turn on the spot
			Action[ActMoveFwd] = 0.f;
	}
	bHasAction = true;
	UE_LOG(LogManipleLyra, VeryVerbose, TEXT("agent %d aim: dist=%.0f visible=%d yaw_err=%.1f pitch_err=%.1f fire=%d fwd_ray=%.2f aim=%s"),
		AgentId, E[3] * PosScale, bVisible, YawErrDeg, PitchErrDeg, Action[ActFire] > 0.f, Obs[ObsRays], *Aim.ToCompactString());
}

void UManipleAgentComponent::ApplyAction(float DeltaTime)
{
	ALyraCharacter* Me = Character.Get();
	AAIController* Ctrl = AI.Get();

	if (UBrainComponent* Brain = Ctrl->FindComponentByClass<UBrainComponent>())
	{
		if (Brain->IsRunning())
			Brain->StopLogic(TEXT("Maniple"));
	}

	Aim.Yaw = FRotator::NormalizeAxis(Aim.Yaw + FMath::Clamp(Action[ActYawRate], -1.f, 1.f) * MaxYawDegPerSec * DeltaTime);
	Aim.Pitch = FMath::Clamp(Aim.Pitch + FMath::Clamp(Action[ActPitchRate], -1.f, 1.f) * MaxPitchDegPerSec * DeltaTime, -80.f, 80.f);
	Aim.Roll = 0.f;
	Ctrl->SetControlRotation(Aim); // before firing: the shot uses the controller's rotation

	const FRotator Frame(0.f, Aim.Yaw, 0.f);
	Me->AddMovementInput(Frame.RotateVector(FVector::ForwardVector), FMath::Clamp(Action[ActMoveFwd], -1.f, 1.f));
	Me->AddMovementInput(Frame.RotateVector(FVector::RightVector), FMath::Clamp(Action[ActMoveRight], -1.f, 1.f));

	UpdateWeapon(DeltaTime); // shot detection, reload, bMagazineEmpty for the fire decision below

	if (ULyraAbilitySystemComponent* ASC = Me->GetLyraAbilitySystemComponent())
	{
		static const FGameplayTag FireTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Fire"));
		const bool bWantFire = Action[ActFire] > 0.f && !bMagazineEmpty;
		if (bWantFire != bFiring)
		{
			bWantFire ? ASC->AbilityInputTagPressed(FireTag) : ASC->AbilityInputTagReleased(FireTag);
			bFiring = bWantFire;
			if (bWantFire)
				UE_LOG(LogManipleLyra, VeryVerbose, TEXT("agent %d fire: nearest=%.0f visible=%d yaw_err=%.1f pitch_err=%.1f vel=%.0f"),
					AgentId, LastNearestDist, bLastVisible ? 1 : 0, LastYawErr, LastPitchErr, Character->GetVelocity().Size());
		}
		ASC->ProcessAbilityInput(DeltaTime, false); // bots have no player controller doing this
	}
}

void UManipleAgentComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (!IsReady())
		return;
	EpisodeTime += DeltaTime;
	if (bHasAction)
		ApplyAction(DeltaTime);
}

// ---------- transitions ----------

void UManipleAgentComponent::BeginTransition(TConstArrayView<float> Obs, TConstArrayView<float> InAction, float LogP, int64 PolicyVersion)
{
	TransObs = TArray<float>(Obs.GetData(), Obs.Num());
	TransAction = TArray<float>(InAction.GetData(), InAction.Num());
	TransLogP = LogP;
	TransVersion = PolicyVersion;
	bHasTransition = true;
	AddReward(RewardStep);
}

void UManipleAgentComponent::AddShapingFromObservation()
{
	if (LastAim > 0.f)
	{
		const float AngleDeg = FMath::RadiansToDegrees(FMath::Acos(FMath::Clamp(LastAim, -1.f, 1.f)));
		const float Linear = FMath::Max(0.f, 1.f - AngleDeg / AimRangeDeg);
		const float Sharp = FMath::Exp(-FMath::Square(AngleDeg / AimPeakDeg));
		AddReward(RewardAimScale * (0.5f * Linear + 0.5f * Sharp));
	}
	if (bApproachShaping && PrevNearestDist > 0.f && LastNearestDist > 0.f)
	{
		const float From = FMath::Max(PrevNearestDist, ApproachStopDist);
		const float To = FMath::Max(LastNearestDist, ApproachStopDist);
		AddReward(RewardApproachScale * (From - To) / PosScale);
	}
	PrevNearestDist = LastNearestDist;
}

void UManipleAgentComponent::FlushTransition(bool bDone)
{
	if (!bHasTransition)
		return;
	bHasTransition = false;
	EpisodeReturn += RewardAccum;
	if (Subsystem.IsValid())
		Subsystem->AddTransition(*this, TransObs, TransAction, RewardAccum, bDone, TransLogP, TransVersion);
	RewardAccum = 0.f;
	if (bDone)
	{
		if (Subsystem.IsValid())
			Subsystem->OnEpisodeEnd(*this);
		++EpisodeId;
		EpisodeReturn = 0.f;
		EpisodeTime = 0.f;
		PrevNearestDist = -1.f;
		if (!bDead)
			RefillAmmo();
	}
}

// ---------- curriculum placement ----------

void UManipleAgentComponent::Place(const FVector& Location, float Yaw)
{
	if (!Character.IsValid())
		return;
	Character->TeleportTo(Location, FRotator(0.f, Yaw, 0.f));
	Aim = FRotator(0.f, Yaw, 0.f);
	if (AI.IsValid())
		AI->SetControlRotation(Aim);
	PrevNearestDist = -1.f;
	bPlaced = true;
}

void UManipleAgentComponent::FaceTowards(const FVector& Target)
{
	if (!Character.IsValid())
		return;
	Aim = FRotator(0.f, (Target - Character->GetActorLocation()).Rotation().Yaw, 0.f);
	if (AI.IsValid())
		AI->SetControlRotation(Aim);
}
