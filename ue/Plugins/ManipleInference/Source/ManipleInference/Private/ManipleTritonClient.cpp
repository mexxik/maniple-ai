#include "ManipleTritonClient.h"
#include "ManipleInference.h"
#include "HttpModule.h"
#include "HttpManager.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformProcess.h"

// ---------- types ----------

FManipleTensor FManipleTensor::MakeFloat(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<float> Values)
{
	FManipleTensor T;
	T.Name = InName;
	T.Shape = TArray<int64>(InShape.GetData(), InShape.Num());
	T.Datatype = TEXT("FP32");
	T.Data.SetNumUninitialized(Values.Num() * sizeof(float));
	FMemory::Memcpy(T.Data.GetData(), Values.GetData(), T.Data.Num());
	return T;
}

int64 FManipleTensor::NumElements() const
{
	int64 N = 1;
	for (int64 D : Shape) N *= D;
	return Shape.Num() ? N : 0;
}

int32 FManipleTensor::ElementSize(const FString& Datatype)
{
	static const TMap<FString, int32> Sizes = {
		{TEXT("BOOL"), 1}, {TEXT("UINT8"), 1}, {TEXT("INT8"), 1}, {TEXT("UINT16"), 2}, {TEXT("INT16"), 2}, {TEXT("FP16"), 2}, {TEXT("BF16"), 2},
		{TEXT("UINT32"), 4}, {TEXT("INT32"), 4}, {TEXT("FP32"), 4}, {TEXT("UINT64"), 8}, {TEXT("INT64"), 8}, {TEXT("FP64"), 8} };
	const int32* S = Sizes.Find(Datatype);
	return S ? *S : 0;
}

const FManipleTensor* FManipleInferResult::FindOutput(const FString& Name) const
{
	return Outputs.FindByPredicate([&](const FManipleTensor& T) { return T.Name == Name; });
}

// ---------- client ----------

FManipleTritonClient::FManipleTritonClient(const FString& InBaseUrl, float InTimeoutSec)
	: BaseUrl(InBaseUrl), TimeoutSec(InTimeoutSec)
{
	while (BaseUrl.EndsWith(TEXT("/"))) BaseUrl.LeftChopInline(1);
}

void FManipleTritonClient::IsServerReady(FManipleReadyComplete OnComplete) const
{
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Req = FHttpModule::Get().CreateRequest();
	Req->SetURL(BaseUrl + TEXT("/v2/health/ready"));
	Req->SetVerb(TEXT("GET"));
	Req->SetTimeout(TimeoutSec);
	Req->OnProcessRequestComplete().BindLambda([OnComplete](FHttpRequestPtr, FHttpResponsePtr Resp, bool bOk)
	{
		OnComplete.ExecuteIfBound(bOk && Resp.IsValid() && Resp->GetResponseCode() == 200);
	});
	Req->ProcessRequest();
}

bool FManipleTritonClient::IsServerReadySync(float WaitSec) const
{
	TSharedRef<bool> Done = MakeShared<bool>(false);
	TSharedRef<bool> Ready = MakeShared<bool>(false);
	IsServerReady(FManipleReadyComplete::CreateLambda([Done, Ready](bool bReady) { *Ready = bReady; *Done = true; }));
	const double Deadline = FPlatformTime::Seconds() + WaitSec;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		FHttpModule::Get().GetHttpManager().Tick(0.01f);
		FPlatformProcess::Sleep(0.005f);
	}
	return *Done && *Ready;
}

TArray<uint8> FManipleTritonClient::BuildInferBody(const TArray<FManipleTensor>& Inputs, const TArray<FString>& OutputNames, int32& OutJsonLength)
{
	TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
	TArray<TSharedPtr<FJsonValue>> InputsJson;
	int64 BinaryBytes = 0;
	for (const FManipleTensor& T : Inputs)
	{
		TSharedRef<FJsonObject> In = MakeShared<FJsonObject>();
		In->SetStringField(TEXT("name"), T.Name);
		In->SetStringField(TEXT("datatype"), T.Datatype);
		TArray<TSharedPtr<FJsonValue>> Shape;
		for (int64 D : T.Shape) Shape.Add(MakeShared<FJsonValueNumber>((double)D));
		In->SetArrayField(TEXT("shape"), Shape);
		TSharedRef<FJsonObject> Params = MakeShared<FJsonObject>();
		Params->SetNumberField(TEXT("binary_data_size"), (double)T.Data.Num());
		In->SetObjectField(TEXT("parameters"), Params);
		InputsJson.Add(MakeShared<FJsonValueObject>(In));
		BinaryBytes += T.Data.Num();
	}
	Root->SetArrayField(TEXT("inputs"), InputsJson);
	if (OutputNames.Num() > 0)
	{
		TArray<TSharedPtr<FJsonValue>> OutsJson;
		for (const FString& N : OutputNames)
		{
			TSharedRef<FJsonObject> Out = MakeShared<FJsonObject>();
			Out->SetStringField(TEXT("name"), N);
			TSharedRef<FJsonObject> Params = MakeShared<FJsonObject>();
			Params->SetBoolField(TEXT("binary_data"), true);
			Out->SetObjectField(TEXT("parameters"), Params);
			OutsJson.Add(MakeShared<FJsonValueObject>(Out));
		}
		Root->SetArrayField(TEXT("outputs"), OutsJson);
	}
	else
	{
		TSharedRef<FJsonObject> Params = MakeShared<FJsonObject>();
		Params->SetBoolField(TEXT("binary_data_output"), true);
		Root->SetObjectField(TEXT("parameters"), Params);
	}

	FString Json;
	TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer = TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Json);
	FJsonSerializer::Serialize(Root, Writer);

	FTCHARToUTF8 Utf8(*Json);
	OutJsonLength = Utf8.Length();
	TArray<uint8> Body;
	Body.Reserve(OutJsonLength + BinaryBytes);
	Body.Append(reinterpret_cast<const uint8*>(Utf8.Get()), OutJsonLength);
	for (const FManipleTensor& T : Inputs) Body.Append(T.Data);
	return Body;
}

bool FManipleTritonClient::ParseInferResponse(const TArray<uint8>& Body, int32 JsonLength, FManipleInferResult& Out)
{
	const int32 JsonBytes = JsonLength < 0 ? Body.Num() : FMath::Min(JsonLength, Body.Num());
	const FString Json = FString(FUTF8ToTCHAR(reinterpret_cast<const ANSICHAR*>(Body.GetData()), JsonBytes));
	TSharedPtr<FJsonObject> Root;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root.IsValid())
	{
		Out.Error = TEXT("response is not valid JSON");
		return false;
	}
	if (Root->HasField(TEXT("error")))
	{
		Out.Error = Root->GetStringField(TEXT("error"));
		return false;
	}
	Root->TryGetStringField(TEXT("model_name"), Out.ModelName);
	Root->TryGetStringField(TEXT("model_version"), Out.ModelVersion);

	const TArray<TSharedPtr<FJsonValue>>* OutputsJson = nullptr;
	if (!Root->TryGetArrayField(TEXT("outputs"), OutputsJson))
	{
		Out.Error = TEXT("response has no outputs");
		return false;
	}
	int32 Cursor = JsonBytes;
	for (const TSharedPtr<FJsonValue>& V : *OutputsJson)
	{
		const TSharedPtr<FJsonObject>& O = V->AsObject();
		FManipleTensor T;
		T.Name = O->GetStringField(TEXT("name"));
		T.Datatype = O->GetStringField(TEXT("datatype"));
		for (const TSharedPtr<FJsonValue>& D : O->GetArrayField(TEXT("shape"))) T.Shape.Add((int64)D->AsNumber());

		const TSharedPtr<FJsonObject>* Params = nullptr;
		double BinSize = -1;
		if (O->TryGetObjectField(TEXT("parameters"), Params)) (*Params)->TryGetNumberField(TEXT("binary_data_size"), BinSize);
		if (BinSize >= 0)
		{
			const int32 N = (int32)BinSize;
			if (Cursor + N > Body.Num())
			{
				Out.Error = FString::Printf(TEXT("binary payload for '%s' truncated"), *T.Name);
				return false;
			}
			T.Data.Append(Body.GetData() + Cursor, N);
			Cursor += N;
		}
		else
		{
			// JSON number array fallback (server ignored binary request)
			const int32 ES = FManipleTensor::ElementSize(T.Datatype);
			if (ES == 0 || T.Datatype != TEXT("FP32"))
			{
				Out.Error = FString::Printf(TEXT("non-binary output '%s' of type %s unsupported"), *T.Name, *T.Datatype);
				return false;
			}
			for (const TSharedPtr<FJsonValue>& D : O->GetArrayField(TEXT("data")))
			{
				const float F = (float)D->AsNumber();
				T.Data.Append(reinterpret_cast<const uint8*>(&F), sizeof(float));
			}
		}
		Out.Outputs.Add(MoveTemp(T));
	}
	return true;
}

void FManipleTritonClient::Infer(const FString& Model, TArray<FManipleTensor> Inputs, FManipleInferComplete OnComplete,
	const TArray<FString>& OutputNames, const FString& Version) const
{
	int32 JsonLen = 0;
	TArray<uint8> Body = BuildInferBody(Inputs, OutputNames, JsonLen);

	FString Url = BaseUrl + TEXT("/v2/models/") + Model;
	if (!Version.IsEmpty()) Url += TEXT("/versions/") + Version;
	Url += TEXT("/infer");

	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Req = FHttpModule::Get().CreateRequest();
	Req->SetURL(Url);
	Req->SetVerb(TEXT("POST"));
	Req->SetTimeout(TimeoutSec);
	Req->SetHeader(TEXT("Content-Type"), TEXT("application/octet-stream"));
	Req->SetHeader(TEXT("Inference-Header-Content-Length"), FString::FromInt(JsonLen));
	Req->SetContent(MoveTemp(Body));

	const double T0 = FPlatformTime::Seconds();
	Req->OnProcessRequestComplete().BindLambda([OnComplete, T0, Model](FHttpRequestPtr, FHttpResponsePtr Resp, bool bOk)
	{
		FManipleInferResult R;
		R.LatencyMs = (FPlatformTime::Seconds() - T0) * 1000.0;
		if (!bOk || !Resp.IsValid())
		{
			R.Error = TEXT("request failed (connection/timeout)");
			OnComplete.ExecuteIfBound(R);
			return;
		}
		R.HttpStatus = Resp->GetResponseCode();
		const FString HeaderLen = Resp->GetHeader(TEXT("Inference-Header-Content-Length"));
		const int32 JsonLength = HeaderLen.IsEmpty() ? -1 : FCString::Atoi(*HeaderLen);
		const bool bParsed = FManipleTritonClient::ParseInferResponse(Resp->GetContent(), JsonLength, R);
		R.bSuccess = bParsed && R.HttpStatus == 200;
		if (!R.bSuccess && R.Error.IsEmpty()) R.Error = FString::Printf(TEXT("HTTP %d"), R.HttpStatus);
		if (!R.bSuccess) UE_LOG(LogManipleInference, Warning, TEXT("infer %s failed: %s"), *Model, *R.Error);
		OnComplete.ExecuteIfBound(R);
	});
	Req->ProcessRequest();
}

FManipleInferResult FManipleTritonClient::InferSync(const FString& Model, TArray<FManipleTensor> Inputs,
	const TArray<FString>& OutputNames, const FString& Version) const
{
	TSharedRef<FManipleInferResult> Result = MakeShared<FManipleInferResult>();
	TSharedRef<bool> Done = MakeShared<bool>(false);
	Infer(Model, MoveTemp(Inputs), FManipleInferComplete::CreateLambda([Result, Done](const FManipleInferResult& R) { *Result = R; *Done = true; }), OutputNames, Version);
	const double Deadline = FPlatformTime::Seconds() + TimeoutSec + 1.0;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		FHttpModule::Get().GetHttpManager().Tick(0.01f);
		FPlatformProcess::Sleep(0.001f);
	}
	if (!*Done) Result->Error = TEXT("InferSync timed out");
	return *Result;
}
