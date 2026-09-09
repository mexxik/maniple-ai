#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ManipleLyraSchema.h"
#include "ManipleAgentComponent.generated.h"

class ALyraCharacter;
class AAIController;
class APlayerState;
class UManipleBotSubsystem;

/**
 * Actuator + sensor for one Lyra bot pawn. Stops the behaviour tree, builds observations on request,
 * applies the latest action every frame, and accumulates reward between decisions. Decisions are driven
 * by UManipleBotSubsystem. One component = one episode: it lives with the pawn, death ends the episode.
 */
UCLASS()
class MANIPLELYRA_API UManipleAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UManipleAgentComponent();

	void Init(int32 InAgentId, UManipleBotSubsystem* InSubsystem);
	int32 GetAgentId() const { return AgentId; }

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	bool IsReady() const; // possessed, alive, can act
	bool IsDead() const { return bDead; }
	int64 GetEpisodeId() const { return EpisodeId; }
	float GetEpisodeTime() const { return EpisodeTime; }

	/** Curriculum role: heuristic bots share the map as opponents and never send transitions. */
	bool bHeuristic = false;
	bool bApproachShaping = false;
	bool bPlaced = false; // curriculum placement done for this life
	int32 PlacedStage = -1; // stage this life was placed in (-1 = Lyra's own spawn)

	/** True while Lyra's warmup immunity (or any other) protects the pawn: nothing scores yet. */
	bool HasDamageImmunity() const;

	/** Teleport for curriculum placement; faces Yaw. */
	void Place(const FVector& Location, float Yaw);
	void FaceTowards(const FVector& Target);
	float GetNearestEnemyDistance() const { return LastNearestDist; }
	void BuildObservation(TArray<float>& Obs) const;
	void SetAction(TConstArrayView<float> InAction);
	void SetRandomAction();
	/** Turn towards the nearest visible enemy in Obs (built by BuildObservation) and fire when aligned. */
	void SetHeuristicAction(TConstArrayView<float> Obs);

	// ---- training: transitions ----

	/** Dense shaping from the observation just built (aim, approach): the outcome of the previous action. Call before FlushTransition. */
	void AddShapingFromObservation();
	/** Remembers what was observed and decided; the reward gathered until the next decision belongs to it. */
	void BeginTransition(TConstArrayView<float> Obs, TConstArrayView<float> InAction, float LogP, int64 PolicyVersion);
	bool HasTransition() const { return bHasTransition; }
	/** Hands the pending transition to the subsystem batch (done = episode over) and resets the reward accumulator. */
	void FlushTransition(bool bDone);

	void AddReward(float R) { RewardAccum += R; }
	float GetEpisodeReturn() const { return EpisodeReturn; }
	int32 Kills = 0, Deaths = 0;
	int32 Shots = 0, Hits = 0; // actual shots (magazine went down) and damage events dealt
	bool IsDry() const { return DrySeconds > 5.f || JamSeconds > 5.f; } // wants to fire but no round has left the gun for a while

	/** Identity helpers for reward attribution: pawn, controller or player state of this agent. */
	bool Owns(const UObject* Obj) const;
	APlayerState* GetPlayerState() const;

private:
	void TakeOverFromBehaviorTree();
	void ApplyAction(float DeltaTime);

	UFUNCTION()
	void OnDeathStarted(AActor* OwningActor);

	// weapon state through Lyra's quick bar (not exported from LyraGame: reached by reflection)
	UObject* GetActiveWeaponItem() const;
	int32 GetMagazineAmmo(UObject* Item) const;
	int32 GetStat(UObject* Item, const TCHAR* Tag) const;
	void RefillAmmo();
	void SetFireInput(class ULyraAbilitySystemComponent& ASC, bool bPressed); // both fire input tags (pistol / rifle & shotgun)
	void UpdateWeapon(float DeltaTime);
	void LogWeaponState() const;
	void LogDryState(const TCHAR* Why) const; // Display: everything about the gun when it will not fire

	int32 AgentId = -1;
	TWeakObjectPtr<UManipleBotSubsystem> Subsystem;
	TWeakObjectPtr<ALyraCharacter> Character;
	TWeakObjectPtr<AAIController> AI;
	float Action[ManipleLyra::ActDim] = {};
	FRotator Aim = FRotator::ZeroRotator; // our view rotation; pushed to the controller every frame (the AI controller resets pitch)
	bool bHasAction = false;
	bool bFiring = false;
	bool bReloading = false; // reload input held this frame (released next frame)
	bool bMagazineEmpty = false; // no fire input until the magazine refills: firing cancels Lyra's reload ability
	float ReloadRetryTimer = 0.f;
	int32 LastMagazine = -1;
	float DrySeconds = 0.f; // fire wanted and magazine empty (or no weapon) for this long
	float StuckSeconds = 0.f; // ... of which with spare ammo available (a reload should have happened) or no item
	bool bDryHealed = false; // refill + reload forced once per dry spell
	float DryLogTimer = 0.f;
	float JamSeconds = 0.f; // fire wanted, rounds in the magazine, nothing fired for this long
	bool bJamHealed = false; // input abilities cancelled once per jam
	float JamLogTimer = 0.f;
	bool bDead = false;
	float DebugTimer = 0.f;
	bool bDebugLogged = false;

	// pending transition
	bool bHasTransition = false;
	TArray<float> TransObs;
	TArray<float> TransAction;
	float TransLogP = 0.f;
	int64 TransVersion = 0;
	float RewardAccum = 0.f;
	float EpisodeReturn = 0.f;
	float EpisodeTime = 0.f;
	int64 EpisodeId = 0;
	mutable float LastAim = 0.f; // from the last BuildObservation: 0..1, 1 = looking straight at a visible enemy
	mutable float LastNearestDist = -1.f; // cm, -1 = no enemy
	mutable float LastYawErr = 0.f, LastPitchErr = 0.f; // deg, aim error to the nearest enemy (debug)
	mutable bool bLastVisible = false;
	float PrevNearestDist = -1.f; // at the previous decision, for approach shaping
};
