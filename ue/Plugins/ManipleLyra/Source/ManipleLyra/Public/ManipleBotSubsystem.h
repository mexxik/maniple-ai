#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "GameplayTagContainer.h"
#include "GameFramework/GameplayMessageSubsystem.h"
#include "ManipleAgentClient.h"
#include "ManipleBotConfig.h"
#include "ManipleBotSubsystem.generated.h"

class FManipleTritonClient;
class UManipleAgentComponent;
class AAIController;
struct FLyraVerbMessage;

/** One curriculum stage: where bots (re)start, who they fight, which shaping is on, when to move on. */
struct FManipleStage
{
	enum class ESpawn : uint8
	{
		Lyra, // Lyra's own player starts
		Pair, // next to a random enemy, MinDist..MaxDist away
		Near // random reachable point within MaxDist of a random enemy
	};

	const TCHAR* Name;
	ESpawn Spawn;
	float MinDist, MaxDist; // cm
	bool bFaceEachOther; // both turn to each other on placement
	bool bRequireLineOfSight;
	EManipleOpponents Opponents;
	bool bApproachShaping;
	float PromoteKillsPerMin; // <= 0: final stage
};

/**
 * Created only when -ManipleBrain is set. Server side: finds Lyra bot controllers, attaches
 * UManipleAgentComponent to their pawns (also after respawn), optionally spawns extra bots, runs the
 * decision loop at DecisionHz (one act request for all policy agents, heuristic opponents locally), and in
 * train mode registers the policy, attributes rewards from Lyra's damage / elimination messages and sends
 * transitions back every tick. The curriculum places bots per stage and advances stages on the kill rate.
 * Logs grep-friendly "bench:" / "train:" lines every 5 s of game time.
 */
UCLASS()
class MANIPLELYRA_API UManipleBotSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UManipleBotSubsystem, STATGROUP_Tickables); }
	virtual bool IsTickable() const override { return Config.IsActive(); }

	const FManipleBotConfig& GetConfig() const { return Config; }

	// ---- called by agents ----
	void AddTransition(const UManipleAgentComponent& Agent, TConstArrayView<float> Obs, TConstArrayView<float> Action, float Reward,
		bool bDone, float LogP, int64 PolicyVersion);
	void OnEpisodeEnd(const UManipleAgentComponent& Agent);
	void OnAgentDied(const UManipleAgentComponent& Agent);
	void OnAgentShot(const UManipleAgentComponent& Agent);

private:
	void OnWorldStarted();
	void ScanForBots();
	void SpawnExtraBots();
	void MakeEndless();
	void Register();
	void Decide();
	void FlushTransitions();
	void UpdateSpectator(float DeltaTime);
	void LogStats();

	// curriculum
	const FManipleStage* CurrentStage() const;
	void ApplyStage(int32 NewStage);
	void AssignRole(UManipleAgentComponent& Agent) const;
	void PlaceAgent(UManipleAgentComponent& Agent);
	const FManipleStage* PickPlacementStage() const; // current stage, or an easier one for a share of placements
	void UpdateCurriculum();

	// reward attribution
	void OnDamageMessage(FGameplayTag Channel, const FLyraVerbMessage& Msg);
	void OnEliminationMessage(FGameplayTag Channel, const FLyraVerbMessage& Msg);
	UManipleAgentComponent* FindAgent(const UObject* Obj) const;

	FManipleBotConfig Config;
	TSharedPtr<FManipleTritonClient> Client;
	TSharedPtr<FManipleAgentClient> Agent;
	FManipleTransitionBatch Transitions;
	bool bRegistered = false;
	bool bRegisterPending = false;
	double RegisterRetryAt = 0.0;
	bool bWorldStarted = false;

	TArray<TWeakObjectPtr<AAIController>> OwnedControllers;
	TArray<TWeakObjectPtr<UManipleAgentComponent>> Agents; // refreshed each scan
	int32 NextAgentId = 0;
	bool bSpawnedExtra = false;
	bool bEndlessApplied = false;
	FGameplayMessageListenerHandle DamageListener, EliminationListener;

	float ScanTimer = 0.f, DecisionTimer = 0.f, StatsTimer = 0.f, SpectateTimer = 0.f;
	int32 SpectateIndex = 0;
	int32 ActInFlight = 0;

	// curriculum state (the stage itself survives map reloads, see the .cpp)
	int32 Stage = -1;
	int32 StageAgeWindows = 0; // stats windows since the stage began
	TArray<int32> KillHistory; // policy kills per stats window, newest last
	TMap<TWeakObjectPtr<UManipleAgentComponent>, int32> PlaceAttempts;

	// stats window
	int32 WinTicks = 0, WinRequests = 0, WinFailures = 0, WinFrames = 0, WinRows = 0, WinEpisodes = 0, WinKills = 0, WinHeuristicKills = 0,
		  WinDeaths = 0, WinActionRows = 0, WinFireRows = 0, WinPlacements = 0, WinShots = 0, WinHits = 0;
	double WinFrameSec = 0.0, WinReturnSum = 0.0, WinRewardSum = 0.0;
	TArray<double> WinTickLatencyMs;
	int64 LastPolicyVersion = 0;
	FString LastTrainStatus;
};
