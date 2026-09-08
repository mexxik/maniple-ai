#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

class FManipleTritonClient;
class FJsonObject;

/**
 * What the game sends with register: the contract shared by game, trainer and exported policy.
 * Mirrors triton/common/maniple/spec.py (AgentSpec). Unset strings/values keep the trainer defaults.
 */
struct MANIPLEINFERENCE_API FManipleAgentSpec
{
	int32 ObsDim = 0;

	// action space: continuous [ActDim] in [Low, High], or discrete with DiscreteN choices
	int32 ActDim = 0;
	int32 DiscreteN = 0;
	float Low = -1.f;
	float High = 1.f;

	// network (net.*): preset auto|small|medium|large|custom, hidden used with custom
	FString Preset = TEXT("auto");
	TArray<int32> Hidden;
	FString Activation = TEXT("tanh");
	bool bLayerNorm = false;
	bool bNormalizeObs = true;
	float LogStdInit = 0.f; // continuous actions: initial log std of the Gaussian (0 = std 1)

	// ppo.* overrides, key -> number (e.g. "lr" -> 3e-4, "rollout" -> 2048); empty = trainer defaults
	TMap<FString, double> Ppo;

	FString ToJson() const;
};

/** Trainer reply to register / observe / status / report / promote / export: the parsed "status" JSON. */
struct MANIPLEINFERENCE_API FManipleAgentStatus
{
	bool bOk = false;
	FString Error;
	TSharedPtr<FJsonObject> Json; // whole status object (spec, versions, buffer, ...), valid when the call returned
	double LatencyMs = 0.0;

	int64 GetInt(const FString& Field, int64 Default = 0) const;
	FString ToString() const;
};

/** Reply to act: one row per observation row. */
struct MANIPLEINFERENCE_API FManipleActResult
{
	bool bOk = false;
	FString Error;
	int32 Rows = 0;
	TArray<float> Action; // [Rows * ActDim] (continuous action, or one-hot for discrete)
	TArray<int64> ActionIndex; // [Rows] (-1 for continuous)
	TArray<float> LogP; // [Rows] behaviour log-prob, send back with observe
	int64 PolicyVersion = 0; // version that produced the actions, send back with observe
	double LatencyMs = 0.0;

	TConstArrayView<float> Row(int32 i, int32 ActDim) const { return TConstArrayView<float>(Action).Slice(i * ActDim, ActDim); }
};

/** Transitions gathered during one decision tick: one row per agent, sent with observe. */
struct MANIPLEINFERENCE_API FManipleTransitionBatch
{
	int32 ObsDim = 0;
	int32 ActDim = 0;
	TArray<float> Obs, Action, Reward, LogP;
	TArray<bool> Done;
	TArray<int64> AgentId, EpisodeId, PolicyVersion;

	FManipleTransitionBatch() = default;
	FManipleTransitionBatch(int32 InObsDim, int32 InActDim)
		: ObsDim(InObsDim)
		, ActDim(InActDim)
	{
	}

	int32 Num() const { return Reward.Num(); }
	void Reset();
	void Add(TConstArrayView<float> InObs, TConstArrayView<float> InAction, float InReward, bool bDone, int64 InAgentId, int64 InEpisodeId,
		int64 InPolicyVersion, float InLogP);
};

/**
 * Client for the Maniple training protocol (the C++ twin of gym/triton_agent.py): one named policy on one
 * Triton server. Training commands go to the algorithm model (ppo_train); act goes to the trainer for channel
 * "latest" (the weights being trained, no export needed) and to the inference model (ppo_infer) otherwise.
 *
 * Game thread only; async completions arrive on the game thread. Sync variants pump until done.
 */
class MANIPLEINFERENCE_API FManipleAgentClient
{
public:
	using FStatusCallback = TFunction<void(const FManipleAgentStatus&)>;
	using FActCallback = TFunction<void(const FManipleActResult&)>;

	FManipleAgentClient(TSharedPtr<FManipleTritonClient> InClient, const FString& InName, const FString& InTrainModel = TEXT("ppo_train"),
		const FString& InInferModel = TEXT("ppo_infer"));

	const FString& GetName() const { return Name; }
	TSharedPtr<FManipleTritonClient> GetClient() const { return Client; }

	// ---- training side (ppo_train) ----
	void Register(const FManipleAgentSpec& Spec, FStatusCallback OnDone = nullptr);
	FManipleAgentStatus RegisterSync(const FManipleAgentSpec& Spec);
	void Observe(const FManipleTransitionBatch& Batch, FStatusCallback OnDone = nullptr);
	void Status(FStatusCallback OnDone);
	FManipleAgentStatus StatusSync();
	void Report(int64 Version, float Score, int64 Episodes, FStatusCallback OnDone = nullptr);
	void Promote(int64 Version, FStatusCallback OnDone = nullptr);
	void Export(FStatusCallback OnDone = nullptr);

	// ---- inference side ----
	/** Obs is [Rows * ObsDim]. Channel: latest | best | stable | "<version>". Training actors use latest + explore. */
	void Act(TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel, FActCallback OnDone);
	FManipleActResult ActSync(TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel);

private:
	void TrainCall(const TCHAR* Command, TArray<FManipleTensor> Extra, FStatusCallback OnDone);
	FManipleAgentStatus TrainCallSync(const TCHAR* Command, TArray<FManipleTensor> Extra);
	static FManipleAgentStatus ParseStatus(const FManipleInferResult& R, const TCHAR* What);

	TSharedPtr<FManipleTritonClient> Client;
	FString Name, TrainModel, InferModel;
};
