#include "ManipleTritonClient.h"
#include "ManipleInference.h"
#include "Triton/ManipleGrpcIncludes.h"
#include "Containers/Queue.h"
#include "Containers/Ticker.h"
#include "HAL/Thread.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformProcess.h"
#include <atomic>
#include <chrono>
#include <memory>

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

// ---------- gRPC plumbing ----------

namespace
{
	/** One in-flight RPC. Owned by the completion queue until it finishes, then by the game thread until delivered. */
	struct FCallBase
	{
		grpc::ClientContext Ctx;
		grpc::Status Status;
		double StartSec = 0.0;
		virtual ~FCallBase() = default;
		virtual void Deliver() = 0;   // game thread
	};

	struct FInferCall final : FCallBase
	{
		inference::ModelInferResponse Response;
		std::unique_ptr<grpc::ClientAsyncResponseReader<inference::ModelInferResponse>> Reader;
		FManipleInferComplete OnComplete;
		FString Model;

		virtual void Deliver() override
		{
			FManipleInferResult R;
			R.LatencyMs = (FPlatformTime::Seconds() - StartSec) * 1000.0;
			R.StatusCode = (int32)Status.error_code();
			if (!Status.ok())
			{
				R.Error = FString::Printf(TEXT("grpc %d: %s"), R.StatusCode, UTF8_TO_TCHAR(Status.error_message().c_str()));
				UE_LOG(LogManipleInference, Warning, TEXT("infer %s failed: %s"), *Model, *R.Error);
				OnComplete.ExecuteIfBound(R);
				return;
			}
			R.ModelName = UTF8_TO_TCHAR(Response.model_name().c_str());
			R.ModelVersion = UTF8_TO_TCHAR(Response.model_version().c_str());
			const int32 N = Response.outputs_size();
			R.Outputs.Reserve(N);
			for (int32 i = 0; i < N; ++i)
			{
				const inference::ModelInferResponse::InferOutputTensor& O = Response.outputs(i);
				FManipleTensor T;
				T.Name = UTF8_TO_TCHAR(O.name().c_str());
				T.Datatype = UTF8_TO_TCHAR(O.datatype().c_str());
				T.Shape.Reserve(O.shape_size());
				for (int32 d = 0; d < O.shape_size(); ++d) T.Shape.Add(O.shape(d));
				if (i < Response.raw_output_contents_size())
				{
					const std::string& Raw = Response.raw_output_contents(i);
					T.Data.Append(reinterpret_cast<const uint8*>(Raw.data()), (int32)Raw.size());
				}
				else if (O.has_contents() && O.contents().fp32_contents_size() > 0)
				{
					T.Data.Append(reinterpret_cast<const uint8*>(O.contents().fp32_contents().data()), O.contents().fp32_contents_size() * sizeof(float));
				}
				R.Outputs.Add(MoveTemp(T));
			}
			R.bSuccess = true;
			OnComplete.ExecuteIfBound(R);
		}
	};

	struct FReadyCall final : FCallBase
	{
		inference::ServerReadyResponse Response;
		std::unique_ptr<grpc::ClientAsyncResponseReader<inference::ServerReadyResponse>> Reader;
		FManipleReadyComplete OnComplete;
		virtual void Deliver() override { OnComplete.ExecuteIfBound(Status.ok() && Response.ready()); }
	};
}

struct FManipleTritonClient::FImpl
{
	std::shared_ptr<grpc::Channel> Channel;
	std::unique_ptr<inference::GRPCInferenceService::Stub> Stub;
	grpc::CompletionQueue CQ;
	TUniquePtr<FThread> Thread;
	TQueue<FCallBase*, EQueueMode::Spsc> Finished;   // CQ thread -> game thread
	std::atomic<int32> Pending{ 0 };
	FTSTicker::FDelegateHandle Ticker;

	void Start(const FString& Target)
	{
		grpc::ChannelArguments Args;
		Args.SetMaxReceiveMessageSize(64 << 20);
		Args.SetMaxSendMessageSize(64 << 20);
		Channel = grpc::CreateCustomChannel(TCHAR_TO_UTF8(*Target), grpc::InsecureChannelCredentials(), Args);
		Stub = inference::GRPCInferenceService::NewStub(Channel);
		Thread = MakeUnique<FThread>(TEXT("ManipleGrpcCQ"), [this]()
		{
			void* Tag = nullptr; bool bOk = false;
			while (CQ.Next(&Tag, &bOk))
			{
				Finished.Enqueue(static_cast<FCallBase*>(Tag));
			}
		});
	}

	void Stop()
	{
		CQ.Shutdown();
		if (Thread.IsValid()) { Thread->Join(); Thread.Reset(); }
		FCallBase* Call = nullptr;
		while (Finished.Dequeue(Call)) delete Call;
	}

	void Pump()
	{
		FCallBase* Call = nullptr;
		while (Finished.Dequeue(Call))
		{
			--Pending;
			Call->Deliver();
			delete Call;
		}
	}
};

FManipleTritonClient::FManipleTritonClient(const FString& InTarget, float InTimeoutSec)
	: Impl(MakeUnique<FImpl>()), Target(InTarget), TimeoutSec(InTimeoutSec)
{
	// accept http://host:port too
	Target.RemoveFromStart(TEXT("http://"));
	Target.RemoveFromStart(TEXT("grpc://"));
	while (Target.EndsWith(TEXT("/"))) Target.LeftChopInline(1);
	Impl->Start(Target);
	Impl->Ticker = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([this](float) { PumpCompletions(); return true; }));
}

FManipleTritonClient::~FManipleTritonClient()
{
	FTSTicker::GetCoreTicker().RemoveTicker(Impl->Ticker);
	Impl->Stop();
}

void FManipleTritonClient::PumpCompletions()
{
	Impl->Pump();
}

int32 FManipleTritonClient::NumPending() const
{
	return Impl->Pending.load();
}

void FManipleTritonClient::IsServerReady(FManipleReadyComplete OnComplete)
{
	FReadyCall* Call = new FReadyCall();
	Call->OnComplete = MoveTemp(OnComplete);
	Call->StartSec = FPlatformTime::Seconds();
	Call->Ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds((int64)(TimeoutSec * 1000)));
	++Impl->Pending;
	inference::ServerReadyRequest Req;
	Call->Reader = Impl->Stub->AsyncServerReady(&Call->Ctx, Req, &Impl->CQ);
	Call->Reader->Finish(&Call->Response, &Call->Status, Call);
}

bool FManipleTritonClient::IsServerReadySync(float WaitSec)
{
	TSharedRef<bool> Done = MakeShared<bool>(false);
	TSharedRef<bool> Ready = MakeShared<bool>(false);
	IsServerReady(FManipleReadyComplete::CreateLambda([Done, Ready](bool bReady) { *Ready = bReady; *Done = true; }));
	const double Deadline = FPlatformTime::Seconds() + WaitSec;
	while (!*Done && FPlatformTime::Seconds() < Deadline) { PumpCompletions(); FPlatformProcess::Sleep(0.0005f); }
	return *Done && *Ready;
}

void FManipleTritonClient::Infer(const FString& Model, TArray<FManipleTensor> Inputs, FManipleInferComplete OnComplete,
	const TArray<FString>& OutputNames, const FString& Version)
{
	inference::ModelInferRequest Req;
	Req.set_model_name(TCHAR_TO_UTF8(*Model));
	if (!Version.IsEmpty()) Req.set_model_version(TCHAR_TO_UTF8(*Version));
	for (const FManipleTensor& T : Inputs)
	{
		inference::ModelInferRequest::InferInputTensor* In = Req.add_inputs();
		In->set_name(TCHAR_TO_UTF8(*T.Name));
		In->set_datatype(TCHAR_TO_UTF8(*T.Datatype));
		for (int64 D : T.Shape) In->add_shape(D);
		Req.add_raw_input_contents(T.Data.GetData(), T.Data.Num());
	}
	for (const FString& N : OutputNames) Req.add_outputs()->set_name(TCHAR_TO_UTF8(*N));

	FInferCall* Call = new FInferCall();
	Call->OnComplete = MoveTemp(OnComplete);
	Call->Model = Model;
	Call->StartSec = FPlatformTime::Seconds();
	Call->Ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds((int64)(TimeoutSec * 1000)));
	++Impl->Pending;
	Call->Reader = Impl->Stub->AsyncModelInfer(&Call->Ctx, Req, &Impl->CQ);
	Call->Reader->Finish(&Call->Response, &Call->Status, Call);
}

FManipleInferResult FManipleTritonClient::InferSync(const FString& Model, TArray<FManipleTensor> Inputs,
	const TArray<FString>& OutputNames, const FString& Version)
{
	TSharedRef<FManipleInferResult> Result = MakeShared<FManipleInferResult>();
	TSharedRef<bool> Done = MakeShared<bool>(false);
	Infer(Model, MoveTemp(Inputs), FManipleInferComplete::CreateLambda([Result, Done](const FManipleInferResult& R) { *Result = R; *Done = true; }), OutputNames, Version);
	const double Deadline = FPlatformTime::Seconds() + TimeoutSec + 1.0;
	while (!*Done && FPlatformTime::Seconds() < Deadline) { PumpCompletions(); FPlatformProcess::Sleep(0.0002f); }
	if (!*Done) Result->Error = TEXT("InferSync timed out");
	return *Result;
}
