#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ManipleLyraSchema.h"
#include "ManipleAgentComponent.generated.h"

class ALyraCharacter;
class AAIController;

/**
 * Actuator + sensor for one Lyra bot pawn. Stops the behaviour tree, builds observations on
 * request and applies the latest action every frame. Decisions are driven by UManipleBotSubsystem.
 */
UCLASS()
class MANIPLELYRA_API UManipleAgentComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UManipleAgentComponent();

	void Init(int32 InAgentId) { AgentId = InAgentId; }
	int32 GetAgentId() const { return AgentId; }

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	bool IsReady() const;              // possessed, alive, can act
	void BuildObservation(TArray<float>& Obs) const;
	void SetAction(TConstArrayView<float> InAction);
	void SetRandomAction();

	/** per-agent request mode: true while a request for this agent is in flight */
	bool bInferPending = false;

private:
	void TakeOverFromBehaviorTree();
	void ApplyAction(float DeltaTime);

	int32 AgentId = -1;
	TWeakObjectPtr<ALyraCharacter> Character;
	TWeakObjectPtr<AAIController> AI;
	float Action[ManipleLyra::ActDim] = {};
	bool bHasAction = false;
	bool bFiring = false;
};
