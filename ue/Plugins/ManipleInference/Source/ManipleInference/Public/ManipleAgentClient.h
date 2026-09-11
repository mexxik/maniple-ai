#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

class FManipleTritonClient;
class FJsonObject;

/**
 * One named observation input of a policy (mirrors InputSpec in triton/common/maniple/spec.py).
 * Names are fixed by the wire: "obs" (FP32 [D]), "frame" (UINT8 [H, W, C]), "audio" (FP32 [S]), "text".
 */
struct MANIPLEINFERENCE_API FManipleInputSpec
{
	FString Name;
	TArray<int32> Shape; // one row, e.g. {32} or {84, 84, 4}
	FString Dtype = TEXT("fp32"); // fp32 | uint8 (fixed per name)
	FString Encoder; // mlp | cnn; empty = the trainer default for the name
	TArray<int32> Hidden; // layers inside the encoder; empty = trainer default

	static FManipleInputSpec Vector(int32 Dim) { return {TEXT("obs"), {Dim}, TEXT("fp32"), TEXT("mlp"), {}}; }
	static FManipleInputSpec Frame(int32 Height, int32 Width, int32 Channels)
	{
		return {TEXT("frame"), {Height, Width, Channels}, TEXT("uint8"), TEXT("cnn"), {}};
	}
	int32 Size() const;
};

/** One action group (mirrors ActionGroup): continuous values in [Low, High], or one discrete choice out of N. */
struct MANIPLEINFERENCE_API FManipleActionGroup
{
	FString Name;
	bool bDiscrete = false;
	int32 Dim = 0; // continuous
	int32 N = 0; // discrete
	float Low = -1.f;
	float High = 1.f;

	static FManipleActionGroup Continuous(const FString& InName, int32 InDim, float InLow = -1.f, float InHigh = 1.f)
	{
		return {InName, false, InDim, 0, InLow, InHigh};
	}
	static FManipleActionGroup Discrete(const FString& InName, int32 InN) { return {InName, true, 0, InN}; }
	/** Width of this group in the flat action row (values, or one-hot). */
	int32 OutDim() const { return bDiscrete ? N : Dim; }
};

/**
 * What the game sends with register: the contract shared by game, trainer and exported policy.
 * Mirrors triton/common/maniple/spec.py (AgentSpec). Unset strings/values keep the trainer defaults.
 *
 * Two ways to describe the policy:
 *   v1  ObsDim + ActDim / DiscreteN: one vector input "obs", one action group "action" (what ManipleLyra sends)
 *   v2  Inputs + Actions: named inputs (vector, frame, ...) and named action groups; used when Inputs is not empty
 * The trainer treats both the same way; v1 is the shorthand for one input and one group.
 */
struct MANIPLEINFERENCE_API FManipleAgentSpec
{
	// ---- v1 shorthand ----
	int32 ObsDim = 0;

	// action space: continuous [ActDim] in [Low, High], or discrete with DiscreteN choices
	int32 ActDim = 0;
	int32 DiscreteN = 0;
	float Low = -1.f;
	float High = 1.f;

	// ---- v2: named inputs and action groups (takes over when Inputs is not empty) ----
	TArray<FManipleInputSpec> Inputs;
	TArray<FManipleActionGroup> Actions;

	// network (net.*): preset auto|small|medium|large|custom|none, hidden used with custom
	FString Preset = TEXT("auto");
	TArray<int32> Hidden;
	FString Activation = TEXT("tanh");
	bool bLayerNorm = false;
	bool bNormalizeObs = true;
	float LogStdInit = 0.f; // continuous actions: initial log std of the Gaussian (0 = std 1)

	// ppo.* overrides, key -> number (e.g. "lr" -> 3e-4, "rollout" -> 2048); empty = trainer defaults
	TMap<FString, double> Ppo;

	// versioning.score: what ranks versions and triggers exports. "return" = mean training episode return (default),
	// "report" = the score this client sends with Report() (a game-defined number, e.g. kills per agent-minute)
	FString ScoreSource;

	bool IsV2() const { return Inputs.Num() > 0; }
	/** Width of the flat action row (ActDim / DiscreteN, or the sum over Actions). */
	int32 ActionDim() const;
	/** Number of discrete groups = columns of action_index. */
	int32 NumDiscreteGroups() const;
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
	TArray<float> Action; // [Rows * ActionDim]: the flat row over all groups (continuous values, one-hot per discrete group)
	TArray<int64> ActionIndex; // [Rows * NumIndexGroups]: chosen index per discrete group
	int32 NumIndexGroups = 0; // discrete groups (columns of ActionIndex); 0 for a continuous-only policy
	TArray<float> LogP; // [Rows] behaviour log-prob, send back with observe
	int64 PolicyVersion = 0; // version that produced the actions, send back with observe
	double LatencyMs = 0.0;

	TConstArrayView<float> Row(int32 i, int32 ActDim) const { return TConstArrayView<float>(Action).Slice(i * ActDim, ActDim); }
	/** Chosen index of discrete group `Group` (0-based, spec order) for row i; -1 if there is none. */
	int64 Index(int32 i, int32 Group = 0) const { return Group < NumIndexGroups ? ActionIndex[i * NumIndexGroups + Group] : -1; }
};

/**
 * Transitions gathered during one decision tick: one row per agent, sent with observe.
 *
 * Two ways to add a row: Add() with one float vector (the v1 "obs" input), or AddRow() with one FManipleTensor
 * per named input (shape = one row, e.g. frame {84, 84, 4} UINT8). A batch uses one of the two.
 */
struct MANIPLEINFERENCE_API FManipleTransitionBatch
{
	int32 ObsDim = 0;
	int32 ActDim = 0;
	TArray<float> Obs, Action, Reward, LogP;
	TArray<bool> Done;
	TArray<int64> AgentId, EpisodeId, PolicyVersion;

	/** Named inputs (AddRow): each column is [Rows, ...RowShape] in Data, appended row by row. */
	struct FColumn
	{
		FString Name;
		FString Datatype;
		TArray<int64> RowShape;
		TArray<uint8> Data;
	};
	TArray<FColumn> Columns;

	FManipleTransitionBatch() = default;
	FManipleTransitionBatch(int32 InObsDim, int32 InActDim)
		: ObsDim(InObsDim)
		, ActDim(InActDim)
	{
	}
	/** Named-input batch: ActionDim = width of the flat action row (FManipleAgentSpec::ActionDim()). */
	static FManipleTransitionBatch Named(int32 InActionDim) { return FManipleTransitionBatch(0, InActionDim); }

	int32 Num() const { return Reward.Num(); }
	void Reset();
	void Add(TConstArrayView<float> InObs, TConstArrayView<float> InAction, float InReward, bool bDone, int64 InAgentId, int64 InEpisodeId,
		int64 InPolicyVersion, float InLogP);
	/** One row with named inputs; every row must carry the same inputs (name, datatype, row shape). */
	void AddRow(TConstArrayView<FManipleTensor> InInputs, TConstArrayView<float> InAction, float InReward, bool bDone, int64 InAgentId,
		int64 InEpisodeId, int64 InPolicyVersion, float InLogP);
	/** The observation tensors of this batch as sent on the wire ([Rows, ...] each). */
	TArray<FManipleTensor> InputTensors() const;

private:
	void AddCommon(TConstArrayView<float> InAction, float InReward, bool bDone, int64 InAgentId, int64 InEpisodeId, int64 InPolicyVersion,
		float InLogP);
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
	/** Obs is [Rows * ObsDim] (the v1 "obs" input). Channel: latest | best | stable | "<version>". Training actors use latest + explore. */
	void Act(TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel, FActCallback OnDone);
	FManipleActResult ActSync(TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel);
	/** Named inputs, each already batched: shape [Rows, ...row shape] (e.g. "frame" [Rows, 84, 84, 4] UINT8). */
	void Act(TArray<FManipleTensor> Inputs, int32 Rows, bool bExplore, const FString& Channel, FActCallback OnDone);
	FManipleActResult ActSync(TArray<FManipleTensor> Inputs, int32 Rows, bool bExplore, const FString& Channel);

private:
	void TrainCall(const TCHAR* Command, TArray<FManipleTensor> Extra, FStatusCallback OnDone);
	FManipleAgentStatus TrainCallSync(const TCHAR* Command, TArray<FManipleTensor> Extra);
	static FManipleAgentStatus ParseStatus(const FManipleInferResult& R, const TCHAR* What);

	TSharedPtr<FManipleTritonClient> Client;
	FString Name, TrainModel, InferModel;
};
