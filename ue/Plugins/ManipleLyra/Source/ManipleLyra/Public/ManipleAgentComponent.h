#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ManipleBotConfig.h"
#include "ManipleLyraSchema.h"
#include "ManipleAgentComponent.generated.h"

class FManipleTritonClient;
class ALyraCharacter;
class AAIController;
class ULyraAbilitySystemComponent;
struct FManipleInferResult;

/**
 * Drives one Lyra bot pawn from a policy instead of its behaviour tree.
 * Attached by UManipleBotSubsystem; builds an observation at DecisionHz, gets an action
 * (random or via Triton), and applies the latest action every frame.
 */
UCLASS()
class MANIPLELYRA_API UManipleAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UManipleAgentComponent();

	void Init(const FManipleBotConfig& InConfig, TSharedPtr<FManipleTritonClient> InClient, int32 InAgentId);

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	int32 GetAgentId() const { return AgentId; }

	// stats (read by the subsystem for periodic logging)
	int32 Decisions = 0;
	int32 Failures = 0;
	double LatencySumMs = 0.0;

private:
	void TakeOverFromBehaviorTree();
	void Decide();
	void BuildObservation(TArray<float>& Obs) const;
	void ApplyAction(float DeltaTime);
	void OnInferComplete(const FManipleInferResult& R);

	FManipleBotConfig Config;
	TSharedPtr<FManipleTritonClient> Client;
	int32 AgentId = -1;

	TWeakObjectPtr<ALyraCharacter> Character;
	TWeakObjectPtr<AAIController> AI;

	float Action[ManipleLyra::ActDim] = {};
	bool bHasAction = false;
	bool bInferPending = false;
	bool bFiring = false;
	float DecisionTimer = 0.f;
	FTimerHandle DecisionTimerHandle;
};
