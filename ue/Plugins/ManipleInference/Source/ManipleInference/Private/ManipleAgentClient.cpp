#include "ManipleAgentClient.h"
#include "ManipleTritonClient.h"
#include "ManipleInference.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformProcess.h"

// ---------- spec ----------

int32 FManipleInputSpec::Size() const
{
	int32 N = 1;
	for (int32 D : Shape)
		N *= D;
	return N;
}

int32 FManipleAgentSpec::ActionDim() const
{
	if (!IsV2())
		return DiscreteN > 0 ? DiscreteN : ActDim;
	int32 N = 0;
	for (const FManipleActionGroup& G : Actions)
		N += G.OutDim();
	return N;
}

int32 FManipleAgentSpec::NumDiscreteGroups() const
{
	if (!IsV2())
		return DiscreteN > 0 ? 1 : 0;
	int32 N = 0;
	for (const FManipleActionGroup& G : Actions)
		N += G.bDiscrete ? 1 : 0;
	return N;
}

static TArray<TSharedPtr<FJsonValue>> IntArray(const TArray<int32>& Values)
{
	TArray<TSharedPtr<FJsonValue>> Out;
	for (int32 V : Values)
		Out.Add(MakeShared<FJsonValueNumber>(V));
	return Out;
}

static TSharedRef<FJsonObject> ActionGroupJson(const FManipleActionGroup& G)
{
	TSharedRef<FJsonObject> Action = MakeShared<FJsonObject>();
	if (G.bDiscrete)
	{
		Action->SetStringField(TEXT("type"), TEXT("discrete"));
		Action->SetNumberField(TEXT("n"), G.N);
	}
	else
	{
		Action->SetStringField(TEXT("type"), TEXT("continuous"));
		Action->SetNumberField(TEXT("dim"), G.Dim);
		Action->SetNumberField(TEXT("low"), G.Low);
		Action->SetNumberField(TEXT("high"), G.High);
	}
	return Action;
}

FString FManipleAgentSpec::ToJson() const
{
	TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();

	if (IsV2())
	{
		// named inputs and action groups; the trainer keeps the order given here (rows of the action tensor)
		TSharedRef<FJsonObject> InputsObj = MakeShared<FJsonObject>();
		for (const FManipleInputSpec& I : Inputs)
		{
			TSharedRef<FJsonObject> In = MakeShared<FJsonObject>();
			In->SetArrayField(TEXT("shape"), IntArray(I.Shape));
			In->SetStringField(TEXT("dtype"), I.Dtype);
			if (!I.Encoder.IsEmpty())
				In->SetStringField(TEXT("encoder"), I.Encoder);
			if (I.Hidden.Num() > 0)
				In->SetArrayField(TEXT("hidden"), IntArray(I.Hidden));
			InputsObj->SetObjectField(I.Name, In);
		}
		Root->SetObjectField(TEXT("inputs"), InputsObj);

		TArray<TSharedPtr<FJsonValue>> Groups; // a list keeps the order whatever the JSON object does
		for (const FManipleActionGroup& G : Actions)
		{
			TSharedRef<FJsonObject> Group = ActionGroupJson(G);
			Group->SetStringField(TEXT("name"), G.Name);
			Groups.Add(MakeShared<FJsonValueObject>(Group));
		}
		Root->SetArrayField(TEXT("actions"), Groups);
	}
	else
	{
		TSharedRef<FJsonObject> Obs = MakeShared<FJsonObject>();
		Obs->SetNumberField(TEXT("dim"), ObsDim);
		Root->SetObjectField(TEXT("obs"), Obs);

		const FManipleActionGroup Single = DiscreteN > 0 ? FManipleActionGroup::Discrete(TEXT("action"), DiscreteN)
														 : FManipleActionGroup::Continuous(TEXT("action"), ActDim, Low, High);
		Root->SetObjectField(TEXT("action"), ActionGroupJson(Single));
	}

	TSharedRef<FJsonObject> Net = MakeShared<FJsonObject>();
	Net->SetStringField(TEXT("preset"), Hidden.Num() > 0 ? TEXT("custom") : Preset);
	if (Hidden.Num() > 0)
	{
		TArray<TSharedPtr<FJsonValue>> H;
		for (int32 W : Hidden)
			H.Add(MakeShared<FJsonValueNumber>(W));
		Net->SetArrayField(TEXT("hidden"), H);
	}
	Net->SetStringField(TEXT("activation"), Activation);
	Net->SetBoolField(TEXT("layernorm"), bLayerNorm);
	Net->SetBoolField(TEXT("normalize_obs"), bNormalizeObs);
	Net->SetNumberField(TEXT("log_std_init"), LogStdInit);
	Root->SetObjectField(TEXT("net"), Net);

	if (!ScoreSource.IsEmpty())
	{
		TSharedRef<FJsonObject> V = MakeShared<FJsonObject>();
		V->SetStringField(TEXT("score"), ScoreSource);
		Root->SetObjectField(TEXT("versioning"), V);
	}

	if (Ppo.Num() > 0)
	{
		TSharedRef<FJsonObject> P = MakeShared<FJsonObject>();
		for (const auto& KV : Ppo)
			P->SetNumberField(KV.Key, KV.Value);
		Root->SetObjectField(TEXT("ppo"), P);
	}

	FString Out;
	const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
	FJsonSerializer::Serialize(Root, Writer);
	return Out;
}

// ---------- status ----------

int64 FManipleAgentStatus::GetInt(const FString& Field, int64 Default) const
{
	double V = 0;
	return Json.IsValid() && Json->TryGetNumberField(Field, V) ? (int64)V : Default;
}

FString FManipleAgentStatus::ToString() const
{
	if (!bOk)
		return TEXT("error: ") + Error;
	FString Out;
	const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
	if (Json.IsValid())
		FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);
	return Out;
}

// ---------- transitions ----------

void FManipleTransitionBatch::Reset()
{
	Obs.Reset();
	Action.Reset();
	Reward.Reset();
	LogP.Reset();
	Done.Reset();
	AgentId.Reset();
	EpisodeId.Reset();
	PolicyVersion.Reset();
	for (FColumn& C : Columns)
		C.Data.Reset();
}

void FManipleTransitionBatch::AddCommon(
	TConstArrayView<float> InAction, float InReward, bool bDone, int64 InAgentId, int64 InEpisodeId, int64 InPolicyVersion, float InLogP)
{
	checkf(InAction.Num() == ActDim, TEXT("transition action size mismatch: %d, expected %d"), InAction.Num(), ActDim);
	Action.Append(InAction.GetData(), InAction.Num());
	Reward.Add(InReward);
	Done.Add(bDone);
	AgentId.Add(InAgentId);
	EpisodeId.Add(InEpisodeId);
	PolicyVersion.Add(InPolicyVersion);
	LogP.Add(InLogP);
}

void FManipleTransitionBatch::Add(TConstArrayView<float> InObs, TConstArrayView<float> InAction, float InReward, bool bDone,
	int64 InAgentId, int64 InEpisodeId, int64 InPolicyVersion, float InLogP)
{
	checkf(Columns.Num() == 0, TEXT("a batch uses Add (one obs vector) or AddRow (named inputs), not both"));
	checkf(InObs.Num() == ObsDim, TEXT("transition obs size mismatch: %d, expected %d"), InObs.Num(), ObsDim);
	Obs.Append(InObs.GetData(), InObs.Num());
	AddCommon(InAction, InReward, bDone, InAgentId, InEpisodeId, InPolicyVersion, InLogP);
}

void FManipleTransitionBatch::AddRow(TConstArrayView<FManipleTensor> InInputs, TConstArrayView<float> InAction, float InReward, bool bDone,
	int64 InAgentId, int64 InEpisodeId, int64 InPolicyVersion, float InLogP)
{
	checkf(Obs.Num() == 0, TEXT("a batch uses Add (one obs vector) or AddRow (named inputs), not both"));
	if (Columns.Num() == 0)
	{
		for (const FManipleTensor& T : InInputs)
			Columns.Add({T.Name, T.Datatype, T.Shape, {}});
	}
	checkf(InInputs.Num() == Columns.Num(), TEXT("every row must carry the same named inputs"));
	for (int32 i = 0; i < InInputs.Num(); ++i)
	{
		const FManipleTensor& T = InInputs[i];
		FColumn& C = Columns[i];
		checkf(T.Name == C.Name && T.Datatype == C.Datatype && T.Shape == C.RowShape, TEXT("input '%s': row layout changed"), *T.Name);
		C.Data.Append(T.Data);
	}
	AddCommon(InAction, InReward, bDone, InAgentId, InEpisodeId, InPolicyVersion, InLogP);
}

TArray<FManipleTensor> FManipleTransitionBatch::InputTensors() const
{
	const int32 N = Num();
	TArray<FManipleTensor> Out;
	if (Columns.Num() == 0)
	{
		const int64 ObsShape[2] = {N, ObsDim};
		Out.Add(FManipleTensor::MakeFloat(TEXT("obs"), ObsShape, Obs));
		return Out;
	}
	for (const FColumn& C : Columns)
	{
		FManipleTensor T;
		T.Name = C.Name;
		T.Datatype = C.Datatype;
		T.Shape.Add(N);
		T.Shape.Append(C.RowShape);
		T.Data = C.Data;
		Out.Add(MoveTemp(T));
	}
	return Out;
}

// ---------- client ----------

FManipleAgentClient::FManipleAgentClient(
	TSharedPtr<FManipleTritonClient> InClient, const FString& InName, const FString& InTrainModel, const FString& InInferModel)
	: Client(InClient)
	, Name(InName)
	, TrainModel(InTrainModel)
	, InferModel(InInferModel)
{
}

FManipleAgentStatus FManipleAgentClient::ParseStatus(const FManipleInferResult& R, const TCHAR* What)
{
	FManipleAgentStatus S;
	S.LatencyMs = R.LatencyMs;
	if (!R.bSuccess)
	{
		S.Error = FString::Printf(TEXT("%s: %s"), What, *R.Error);
		return S;
	}
	const FManipleTensor* T = R.FindOutput(TEXT("status"));
	const TArray<FString> Strings = T ? T->AsStrings() : TArray<FString>();
	if (Strings.Num() == 0)
	{
		S.Error = FString::Printf(TEXT("%s: no status output"), What);
		return S;
	}
	TSharedPtr<FJsonObject> Obj;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Strings[0]), Obj) || !Obj.IsValid())
	{
		S.Error = FString::Printf(TEXT("%s: bad status json: %s"), What, *Strings[0]);
		return S;
	}
	S.Json = Obj;
	S.bOk = Obj->HasField(TEXT("ok")) ? Obj->GetBoolField(TEXT("ok")) : true;
	if (!S.bOk)
		S.Error = FString::Printf(TEXT("%s: %s"), What, *Obj->GetStringField(TEXT("error")));
	return S;
}

void FManipleAgentClient::TrainCall(const TCHAR* Command, TArray<FManipleTensor> Extra, FStatusCallback OnDone)
{
	TArray<FManipleTensor> Inputs;
	Inputs.Add(FManipleTensor::MakeString(TEXT("name"), Name));
	Inputs.Add(FManipleTensor::MakeString(TEXT("command"), Command));
	Inputs.Append(MoveTemp(Extra));

	const FString What = FString::Printf(TEXT("%s %s"), *TrainModel, Command);
	Client->Infer(TrainModel, MoveTemp(Inputs),
		FManipleInferComplete::CreateLambda(
			[What, OnDone](const FManipleInferResult& R)
			{
				const FManipleAgentStatus S = ParseStatus(R, *What);
				if (!S.bOk)
					UE_LOG(LogManipleInference, Warning, TEXT("%s"), *S.Error);
				if (OnDone)
					OnDone(S);
			}),
		{TEXT("status")});
}

FManipleAgentStatus FManipleAgentClient::TrainCallSync(const TCHAR* Command, TArray<FManipleTensor> Extra)
{
	TSharedRef<FManipleAgentStatus> Result = MakeShared<FManipleAgentStatus>();
	TSharedRef<bool> Done = MakeShared<bool>(false);
	TrainCall(Command, MoveTemp(Extra),
		[Result, Done](const FManipleAgentStatus& S)
		{
			*Result = S;
			*Done = true;
		});
	const double Deadline = FPlatformTime::Seconds() + 10.0;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		Client->PumpCompletions();
		FPlatformProcess::Sleep(0.0005f);
	}
	if (!*Done)
		Result->Error = FString::Printf(TEXT("%s %s: timed out"), *TrainModel, Command);
	return *Result;
}

void FManipleAgentClient::Register(const FManipleAgentSpec& Spec, FStatusCallback OnDone)
{
	TrainCall(TEXT("register"), {FManipleTensor::MakeString(TEXT("spec"), Spec.ToJson())}, MoveTemp(OnDone));
}

FManipleAgentStatus FManipleAgentClient::RegisterSync(const FManipleAgentSpec& Spec)
{
	return TrainCallSync(TEXT("register"), {FManipleTensor::MakeString(TEXT("spec"), Spec.ToJson())});
}

void FManipleAgentClient::Observe(const FManipleTransitionBatch& B, FStatusCallback OnDone)
{
	const int32 N = B.Num();
	if (N == 0)
		return;
	const int64 ActShape[2] = {N, B.ActDim};
	const int64 Flat[1] = {N};
	TArray<FManipleTensor> Inputs = B.InputTensors();
	Inputs.Add(FManipleTensor::MakeFloat(TEXT("action"), ActShape, B.Action));
	Inputs.Add(FManipleTensor::MakeFloat(TEXT("reward"), Flat, B.Reward));
	Inputs.Add(FManipleTensor::MakeBool(TEXT("done"), Flat, B.Done));
	Inputs.Add(FManipleTensor::MakeInt64(TEXT("agent_id"), Flat, B.AgentId));
	Inputs.Add(FManipleTensor::MakeInt64(TEXT("episode_id"), Flat, B.EpisodeId));
	Inputs.Add(FManipleTensor::MakeInt64(TEXT("policy_version"), Flat, B.PolicyVersion));
	Inputs.Add(FManipleTensor::MakeFloat(TEXT("logp"), Flat, B.LogP));
	TrainCall(TEXT("observe"), MoveTemp(Inputs), MoveTemp(OnDone));
}

void FManipleAgentClient::Status(FStatusCallback OnDone)
{
	TrainCall(TEXT("status"), {}, MoveTemp(OnDone));
}

FManipleAgentStatus FManipleAgentClient::StatusSync()
{
	return TrainCallSync(TEXT("status"), {});
}

void FManipleAgentClient::Report(int64 Version, float Score, int64 Episodes, FStatusCallback OnDone)
{
	const int64 One[1] = {1};
	TrainCall(TEXT("report"),
		{FManipleTensor::MakeInt64(TEXT("version"), One, {Version}), FManipleTensor::MakeFloat(TEXT("score"), One, {Score}),
			FManipleTensor::MakeInt64(TEXT("episodes"), One, {Episodes})},
		MoveTemp(OnDone));
}

void FManipleAgentClient::Promote(int64 Version, FStatusCallback OnDone)
{
	const int64 One[1] = {1};
	TrainCall(TEXT("promote"), {FManipleTensor::MakeInt64(TEXT("version"), One, {Version})}, MoveTemp(OnDone));
}

void FManipleAgentClient::Export(FStatusCallback OnDone)
{
	TrainCall(TEXT("export"), {}, MoveTemp(OnDone));
}

void FManipleAgentClient::Act(
	TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel, FActCallback OnDone)
{
	checkf(Obs.Num() == Rows * ObsDim, TEXT("act: obs has %d floats, expected %d x %d"), Obs.Num(), Rows, ObsDim);
	const int64 ObsShape[2] = {Rows, ObsDim};
	TArray<FManipleTensor> Inputs;
	Inputs.Add(FManipleTensor::MakeFloat(TEXT("obs"), ObsShape, Obs));
	Act(MoveTemp(Inputs), Rows, bExplore, Channel, MoveTemp(OnDone));
}

void FManipleAgentClient::Act(TArray<FManipleTensor> Inputs, int32 Rows, bool bExplore, const FString& Channel, FActCallback OnDone)
{
	for (const FManipleTensor& T : Inputs)
		checkf(T.Shape.Num() > 0 && T.Shape[0] == Rows, TEXT("act: input '%s' has %lld rows, expected %d"), *T.Name,
			T.Shape.Num() > 0 ? T.Shape[0] : 0, Rows);
	const bool bLatest = Channel.IsEmpty() || Channel.Equals(TEXT("latest"), ESearchCase::IgnoreCase);
	const FString Model = bLatest ? TrainModel : InferModel;

	const int64 One[1] = {1};
	Inputs.Insert(FManipleTensor::MakeString(TEXT("name"), Name), 0);
	if (bLatest)
		Inputs.Insert(FManipleTensor::MakeString(TEXT("command"), TEXT("act")), 1);
	else
		Inputs.Insert(FManipleTensor::MakeString(TEXT("channel"), Channel), 1);
	Inputs.Add(FManipleTensor::MakeBool(TEXT("explore"), One, {bExplore}));

	const FString What = Model + TEXT(" act");
	Client->Infer(Model, MoveTemp(Inputs),
		FManipleInferComplete::CreateLambda(
			[What, Rows, OnDone](const FManipleInferResult& R)
			{
				FManipleActResult A;
				A.Rows = Rows;
				A.LatencyMs = R.LatencyMs;
				const FManipleAgentStatus S = ParseStatus(R, *What);
				if (!S.bOk)
				{
					A.Error = S.Error;
					UE_LOG(LogManipleInference, Warning, TEXT("%s"), *A.Error);
					OnDone(A);
					return;
				}
				const FManipleTensor* Action = R.FindOutput(TEXT("action"));
				const FManipleTensor* Index = R.FindOutput(TEXT("action_index"));
				const FManipleTensor* LogP = R.FindOutput(TEXT("logp"));
				const FManipleTensor* Version = R.FindOutput(TEXT("policy_version"));
				if (!Action || Action->Shape.Num() != 2 || Action->Shape[0] != Rows)
				{
					A.Error = What + TEXT(": bad action output");
					UE_LOG(LogManipleInference, Warning, TEXT("%s"), *A.Error);
					OnDone(A);
					return;
				}
				A.Action = TArray<float>(Action->AsFloats());
				if (Index)
				{
					A.ActionIndex = TArray<int64>(Index->AsInt64s());
					A.NumIndexGroups = Index->Shape.Num() == 2 ? (int32)Index->Shape[1] : (Rows > 0 ? A.ActionIndex.Num() / Rows : 0);
				}
				if (LogP)
					A.LogP = TArray<float>(LogP->AsFloats());
				if (Version && Version->AsInt64s().Num() > 0)
					A.PolicyVersion = Version->AsInt64s()[0];
				A.bOk = true;
				OnDone(A);
			}),
		{TEXT("action"), TEXT("action_index"), TEXT("logp"), TEXT("policy_version"), TEXT("status")});
}

FManipleActResult FManipleAgentClient::ActSync(TConstArrayView<float> Obs, int32 Rows, int32 ObsDim, bool bExplore, const FString& Channel)
{
	checkf(Obs.Num() == Rows * ObsDim, TEXT("act: obs has %d floats, expected %d x %d"), Obs.Num(), Rows, ObsDim);
	const int64 ObsShape[2] = {Rows, ObsDim};
	TArray<FManipleTensor> Inputs;
	Inputs.Add(FManipleTensor::MakeFloat(TEXT("obs"), ObsShape, Obs));
	return ActSync(MoveTemp(Inputs), Rows, bExplore, Channel);
}

FManipleActResult FManipleAgentClient::ActSync(TArray<FManipleTensor> Inputs, int32 Rows, bool bExplore, const FString& Channel)
{
	TSharedRef<FManipleActResult> Result = MakeShared<FManipleActResult>();
	TSharedRef<bool> Done = MakeShared<bool>(false);
	Act(MoveTemp(Inputs), Rows, bExplore, Channel,
		[Result, Done](const FManipleActResult& A)
		{
			*Result = A;
			*Done = true;
		});
	const double Deadline = FPlatformTime::Seconds() + 10.0;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		Client->PumpCompletions();
		FPlatformProcess::Sleep(0.0005f);
	}
	if (!*Done)
		Result->Error = TEXT("act: timed out");
	return *Result;
}
