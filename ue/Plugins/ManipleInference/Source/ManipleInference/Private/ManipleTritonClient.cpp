#include "ManipleTritonClient.h"
#include "ManipleInference.h"
#include "Triton/ManipleGrpcIncludes.h"
#include "Containers/Queue.h"
#include "Containers/Ticker.h"
#include "HAL/Thread.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformProcess.h"
#include "HAL/CriticalSection.h"
#include "Misc/ScopeLock.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

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

FManipleTensor FManipleTensor::MakeInt64(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<int64> Values)
{
	FManipleTensor T;
	T.Name = InName;
	T.Shape = TArray<int64>(InShape.GetData(), InShape.Num());
	T.Datatype = TEXT("INT64");
	T.Data.SetNumUninitialized(Values.Num() * sizeof(int64));
	FMemory::Memcpy(T.Data.GetData(), Values.GetData(), T.Data.Num());
	return T;
}

FManipleTensor FManipleTensor::MakeBool(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<bool> Values)
{
	FManipleTensor T;
	T.Name = InName;
	T.Shape = TArray<int64>(InShape.GetData(), InShape.Num());
	T.Datatype = TEXT("BOOL");
	T.Data.SetNumUninitialized(Values.Num());
	for (int32 i = 0; i < Values.Num(); ++i)
		T.Data[i] = Values[i] ? 1 : 0;
	return T;
}

FManipleTensor FManipleTensor::MakeUInt8(const FString& InName, TConstArrayView<int64> InShape, TConstArrayView<uint8> Values)
{
	FManipleTensor T;
	T.Name = InName;
	T.Shape = TArray<int64>(InShape.GetData(), InShape.Num());
	T.Datatype = TEXT("UINT8");
	T.Data = TArray<uint8>(Values.GetData(), Values.Num());
	return T;
}

FManipleTensor FManipleTensor::MakeStrings(const FString& InName, TConstArrayView<FString> Strings)
{
	FManipleTensor T;
	T.Name = InName;
	T.Shape = {Strings.Num()};
	T.Datatype = TEXT("BYTES");
	for (const FString& S : Strings)
	{
		const FTCHARToUTF8 Utf8(*S);
		const uint32 Len = (uint32)Utf8.Length();
		T.Data.Append(reinterpret_cast<const uint8*>(&Len), sizeof(Len));
		T.Data.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Len);
	}
	return T;
}

TArray<FString> FManipleTensor::AsStrings() const
{
	TArray<FString> Out;
	if (Datatype != TEXT("BYTES"))
		return Out;
	int32 Pos = 0;
	while (Pos + 4 <= Data.Num())
	{
		uint32 Len = 0;
		FMemory::Memcpy(&Len, Data.GetData() + Pos, 4);
		Pos += 4;
		if (Pos + (int32)Len > Data.Num())
			break;
		Out.Add(FString(FUTF8ToTCHAR(reinterpret_cast<const ANSICHAR*>(Data.GetData() + Pos), Len)));
		Pos += Len;
	}
	return Out;
}

int64 FManipleTensor::NumElements() const
{
	int64 N = 1;
	for (int64 D : Shape)
		N *= D;
	return Shape.Num() ? N : 0;
}

int32 FManipleTensor::ElementSize(const FString& Datatype)
{
	static const TMap<FString, int32> Sizes = {{TEXT("BOOL"), 1}, {TEXT("UINT8"), 1}, {TEXT("INT8"), 1}, {TEXT("UINT16"), 2},
		{TEXT("INT16"), 2}, {TEXT("FP16"), 2}, {TEXT("BF16"), 2}, {TEXT("UINT32"), 4}, {TEXT("INT32"), 4}, {TEXT("FP32"), 4},
		{TEXT("UINT64"), 8}, {TEXT("INT64"), 8}, {TEXT("FP64"), 8}};
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
	void ToResult(const inference::ModelInferResponse& Response, FManipleInferResult& R)
	{
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
			for (int32 d = 0; d < O.shape_size(); ++d)
				T.Shape.Add(O.shape(d));
			if (i < Response.raw_output_contents_size())
			{
				const std::string& Raw = Response.raw_output_contents(i);
				T.Data.Append(reinterpret_cast<const uint8*>(Raw.data()), (int32)Raw.size());
			}
			else if (O.has_contents() && O.contents().fp32_contents_size() > 0)
			{
				T.Data.Append(
					reinterpret_cast<const uint8*>(O.contents().fp32_contents().data()), O.contents().fp32_contents_size() * sizeof(float));
			}
			R.Outputs.Add(MoveTemp(T));
		}
		R.bSuccess = true;
	}

	struct FImplAccess;

	/** Everything the completion queue hands back is one of these. */
	struct FTag
	{
		virtual ~FTag() = default;
		virtual void OnCompletion(bool bOk, FImplAccess& Impl) = 0; // CQ thread
	};

	/** Unary ServerReady. */
	struct FReadyCall final : FTag
	{
		grpc::ClientContext Ctx;
		grpc::Status Status;
		inference::ServerReadyResponse Response;
		std::unique_ptr<grpc::ClientAsyncResponseReader<inference::ServerReadyResponse>> Reader;
		FManipleReadyComplete OnComplete;
		virtual void OnCompletion(bool bOk, FImplAccess& Impl) override;
	};

	/** One inference request travelling on the stream. */
	struct FStreamRequest
	{
		uint64 Id = 0;
		inference::ModelInferRequest Request;
		FManipleInferComplete OnComplete;
		FString Model;
		double StartSec = 0.0;
	};

	/** Something to hand to the game thread. */
	struct FDelivery
	{
		TUniquePtr<FStreamRequest> Req;
		FManipleInferResult Result;
		TFunction<void()> Other; // for non-inference callbacks (ready)
	};
}

namespace
{
	struct FImplAccess
	{
		FManipleTritonClient::FImpl& I;
	};
}

struct FManipleTritonClient::FImpl
{
	// ---- stream state (one persistent ModelStreamInfer) ----
	struct FStream;
	struct FStreamTag final : FTag
	{
		enum EOp : uint8
		{
			Start,
			Write,
			Read,
			Finish
		} Op;
		FStream* Stream;
		FStreamTag(EOp InOp, FStream* InStream)
			: Op(InOp)
			, Stream(InStream)
		{
		}
		virtual void OnCompletion(bool bOk, FImplAccess& Impl) override;
	};
	struct FStream
	{
		grpc::ClientContext Ctx;
		std::unique_ptr<grpc::ClientAsyncReaderWriter<inference::ModelInferRequest, inference::ModelStreamInferResponse>> RW;
		inference::ModelStreamInferResponse ReadBuf;
		grpc::Status Status;
		std::atomic<bool> bReady{false};
		std::atomic<bool> bWriting{false};
		std::atomic<bool> bDone{false};
		FStreamTag StartTag{FStreamTag::Start, this}, WriteTag{FStreamTag::Write, this}, ReadTag{FStreamTag::Read, this},
			FinishTag{FStreamTag::Finish, this};
	};

	std::shared_ptr<grpc::Channel> Channel;
	std::unique_ptr<inference::GRPCInferenceService::Stub> Stub;
	grpc::CompletionQueue CQ;
	TUniquePtr<FThread> Thread;
	TQueue<FDelivery*, EQueueMode::Mpsc> Finished; // -> game thread
	std::atomic<int32> Pending{0};
	std::atomic<bool> bShuttingDown{false};
	FTSTicker::FDelegateHandle Ticker;
	float TimeoutSec = 5.f;
	FString Target;

	FCriticalSection Lock; // guards everything below
	TUniquePtr<FStream> Stream;
	TArray<TUniquePtr<FStreamRequest>> WriteQueue; // not yet written
	TMap<uint64, TUniquePtr<FStreamRequest>> InFlight; // written, awaiting response
	uint64 NextId = 1;
	double StreamFailedSec = 0.0;
	int32 Reconnects = 0;

	void Start(const FString& InTarget)
	{
		Target = InTarget;
		grpc::ChannelArguments Args;
		Args.SetMaxReceiveMessageSize(64 << 20);
		Args.SetMaxSendMessageSize(64 << 20);
		Args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
		Args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
		Args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
		Channel = grpc::CreateCustomChannel(TCHAR_TO_UTF8(*Target), grpc::InsecureChannelCredentials(), Args);
		Stub = inference::GRPCInferenceService::NewStub(Channel);
		Thread = MakeUnique<FThread>(TEXT("ManipleGrpcCQ"),
			[this]()
			{
				void* Tag = nullptr;
				bool bOk = false;
				FImplAccess Access{*this};
				while (CQ.Next(&Tag, &bOk))
				{
					static_cast<FTag*>(Tag)->OnCompletion(bOk, Access);
				}
			});
		OpenStream();
	}

	void OpenStream()
	{
		FScopeLock L(&Lock);
		Stream = MakeUnique<FStream>();
		Stream->RW = Stub->AsyncModelStreamInfer(&Stream->Ctx, &CQ, &Stream->StartTag);
	}

	void Stop()
	{
		bShuttingDown = true;
		{
			FScopeLock L(&Lock);
			if (Stream.IsValid())
				Stream->Ctx.TryCancel();
		}
		CQ.Shutdown();
		if (Thread.IsValid())
		{
			Thread->Join();
			Thread.Reset();
		}
		FDelivery* D = nullptr;
		while (Finished.Dequeue(D))
			delete D;
		FScopeLock L(&Lock);
		Stream.Reset();
		WriteQueue.Empty();
		InFlight.Empty();
	}

	// --- CQ thread ---
	void KickWrite() // Lock held
	{
		if (!Stream.IsValid() || !Stream->bReady || Stream->bDone || Stream->bWriting || WriteQueue.Num() == 0)
			return;
		TUniquePtr<FStreamRequest> Req = MoveTemp(WriteQueue[0]);
		WriteQueue.RemoveAt(0);
		const inference::ModelInferRequest* Msg = &Req->Request;
		InFlight.Add(Req->Id, MoveTemp(Req));
		Stream->bWriting = true;
		Stream->RW->Write(*Msg, &Stream->WriteTag);
	}

	void FailStream(const FString& Why) // CQ thread
	{
		TArray<TUniquePtr<FStreamRequest>> Failed;
		{
			FScopeLock L(&Lock);
			if (Stream.IsValid())
				Stream->bDone = true;
			for (auto& KV : InFlight)
				Failed.Add(MoveTemp(KV.Value));
			InFlight.Empty();
			StreamFailedSec = FPlatformTime::Seconds();
		}
		for (TUniquePtr<FStreamRequest>& R : Failed)
		{
			FDelivery* D = new FDelivery();
			D->Result.LatencyMs = (FPlatformTime::Seconds() - R->StartSec) * 1000.0;
			D->Result.StatusCode = (int32)grpc::StatusCode::UNAVAILABLE;
			D->Result.Error = TEXT("stream failed: ") + Why;
			D->Req = MoveTemp(R);
			Finished.Enqueue(D);
		}
		if (bShuttingDown)
		{
			UE_LOG(LogManipleInference, Verbose, TEXT("triton stream to %s closed on shutdown"), *Target);
		}
		else
		{
			UE_LOG(LogManipleInference, Warning, TEXT("triton stream to %s failed: %s (%d requests failed)"), *Target, *Why, Failed.Num());
		}
	}

	void OnRead(inference::ModelStreamInferResponse& Msg) // CQ thread
	{
		TUniquePtr<FStreamRequest> Req;
		const std::string& IdStr = Msg.infer_response().id();
		const uint64 Id = IdStr.empty() ? 0 : (uint64)std::stoull(IdStr);
		{
			FScopeLock L(&Lock);
			InFlight.RemoveAndCopyValue(Id, Req);
		}
		if (!Req.IsValid())
		{
			UE_LOG(LogManipleInference, Verbose, TEXT("stream response for unknown id %llu dropped (timed out?)"), Id);
			return;
		}
		FDelivery* D = new FDelivery();
		D->Result.LatencyMs = (FPlatformTime::Seconds() - Req->StartSec) * 1000.0;
		if (!Msg.error_message().empty())
		{
			D->Result.StatusCode = (int32)grpc::StatusCode::INTERNAL;
			D->Result.Error = UTF8_TO_TCHAR(Msg.error_message().c_str());
		}
		else
		{
			ToResult(Msg.infer_response(), D->Result);
		}
		D->Req = MoveTemp(Req);
		Finished.Enqueue(D);
	}

	// --- game thread ---
	void Pump(float NowTimeoutSec)
	{
		FDelivery* D = nullptr;
		while (Finished.Dequeue(D))
		{
			if (D->Req.IsValid())
			{
				--Pending;
				if (!D->Result.bSuccess && D->Result.Error.IsEmpty())
					D->Result.Error = TEXT("unknown error");
				D->Req->OnComplete.ExecuteIfBound(D->Result);
			}
			else if (D->Other)
			{
				--Pending;
				D->Other();
			}
			delete D;
		}

		// timeouts + reconnect
		TArray<TUniquePtr<FStreamRequest>> TimedOut;
		bool bReopen = false;
		{
			FScopeLock L(&Lock);
			const double Now = FPlatformTime::Seconds();
			for (auto It = InFlight.CreateIterator(); It; ++It)
			{
				if (Now - It->Value->StartSec > NowTimeoutSec)
				{
					TimedOut.Add(MoveTemp(It->Value));
					It.RemoveCurrent();
				}
			}
			for (int32 i = WriteQueue.Num() - 1; i >= 0; --i)
			{
				if (Now - WriteQueue[i]->StartSec > NowTimeoutSec)
				{
					TimedOut.Add(MoveTemp(WriteQueue[i]));
					WriteQueue.RemoveAt(i);
				}
			}
			if ((!Stream.IsValid() || Stream->bDone) && !bShuttingDown && Now - StreamFailedSec > 1.0)
				bReopen = true;
		}
		for (TUniquePtr<FStreamRequest>& R : TimedOut)
		{
			--Pending;
			FManipleInferResult Res;
			Res.LatencyMs = NowTimeoutSec * 1000.0;
			Res.StatusCode = (int32)grpc::StatusCode::DEADLINE_EXCEEDED;
			Res.Error = TEXT("timed out");
			R->OnComplete.ExecuteIfBound(Res);
		}
		if (bReopen)
		{
			++Reconnects;
			UE_LOG(LogManipleInference, Display, TEXT("reopening triton stream to %s (attempt %d)"), *Target, Reconnects);
			OpenStream();
		}
	}
};

// ----- completion handlers (CQ thread) -----

namespace
{
	void FReadyCall::OnCompletion(bool bOk, FImplAccess& Impl)
	{
		const bool bReady = bOk && Status.ok() && Response.ready();
		FDelivery* D = new FDelivery();
		FManipleReadyComplete Cb = MoveTemp(OnComplete);
		D->Other = [Cb, bReady]()
		{
			Cb.ExecuteIfBound(bReady);
		};
		Impl.I.Finished.Enqueue(D);
		delete this;
	}
}

void FManipleTritonClient::FImpl::FStreamTag::OnCompletion(bool bOk, FImplAccess& Impl)
{
	FManipleTritonClient::FImpl& I = Impl.I;
	FStream* S = Stream;
	switch (Op)
	{
	case Start:
		if (!bOk)
		{
			I.FailStream(TEXT("could not open stream"));
			return;
		}
		{
			FScopeLock L(&I.Lock);
			S->bReady = true;
			S->RW->Read(&S->ReadBuf, &S->ReadTag);
			I.KickWrite();
		}
		UE_LOG(LogManipleInference, Display, TEXT("triton stream open: %s"), *I.Target);
		return;
	case Write:
		if (!bOk)
		{
			I.FailStream(TEXT("write failed"));
			return;
		}
		{
			FScopeLock L(&I.Lock);
			S->bWriting = false;
			I.KickWrite();
		}
		return;
	case Read:
		if (!bOk)
		{
			// server closed the stream: collect the status, then fail
			if (!S->bDone)
			{
				S->RW->Finish(&S->Status, &S->FinishTag);
			}
			return;
		}
		I.OnRead(S->ReadBuf);
		S->ReadBuf.Clear();
		if (!S->bDone)
			S->RW->Read(&S->ReadBuf, &S->ReadTag);
		return;
	case Finish:
		I.FailStream(S->Status.ok()
				? TEXT("closed by server")
				: FString::Printf(TEXT("grpc %d: %s"), (int32)S->Status.error_code(), UTF8_TO_TCHAR(S->Status.error_message().c_str())));
		return;
	}
}

// ----- public API -----

FManipleTritonClient::FManipleTritonClient(const FString& InTarget, float InTimeoutSec)
	: Impl(MakeUnique<FImpl>())
	, Target(InTarget)
	, TimeoutSec(InTimeoutSec)
{
	Target.RemoveFromStart(TEXT("http://"));
	Target.RemoveFromStart(TEXT("grpc://"));
	while (Target.EndsWith(TEXT("/")))
		Target.LeftChopInline(1);
	Impl->TimeoutSec = TimeoutSec;
	Impl->Start(Target);
	Impl->Ticker = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda(
		[this](float)
		{
			PumpCompletions();
			return true;
		}));
}

FManipleTritonClient::~FManipleTritonClient()
{
	FTSTicker::GetCoreTicker().RemoveTicker(Impl->Ticker);
	Impl->Stop();
}

void FManipleTritonClient::PumpCompletions()
{
	Impl->Pump(TimeoutSec);
}
int32 FManipleTritonClient::NumPending() const
{
	return Impl->Pending.load();
}

bool FManipleTritonClient::IsStreamConnected() const
{
	FScopeLock L(&Impl->Lock);
	return Impl->Stream.IsValid() && Impl->Stream->bReady && !Impl->Stream->bDone;
}

void FManipleTritonClient::IsServerReady(FManipleReadyComplete OnComplete)
{
	FReadyCall* Call = new FReadyCall();
	Call->OnComplete = MoveTemp(OnComplete);
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
	IsServerReady(FManipleReadyComplete::CreateLambda(
		[Done, Ready](bool bReady)
		{
			*Ready = bReady;
			*Done = true;
		}));
	const double Deadline = FPlatformTime::Seconds() + WaitSec;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		PumpCompletions();
		FPlatformProcess::Sleep(0.0005f);
	}
	return *Done && *Ready;
}

void FManipleTritonClient::Infer(const FString& Model, TArray<FManipleTensor> Inputs, FManipleInferComplete OnComplete,
	const TArray<FString>& OutputNames, const FString& Version)
{
	TUniquePtr<FStreamRequest> Req = MakeUnique<FStreamRequest>();
	Req->OnComplete = MoveTemp(OnComplete);
	Req->Model = Model;
	Req->StartSec = FPlatformTime::Seconds();
	inference::ModelInferRequest& M = Req->Request;
	M.set_model_name(TCHAR_TO_UTF8(*Model));
	if (!Version.IsEmpty())
		M.set_model_version(TCHAR_TO_UTF8(*Version));
	for (const FManipleTensor& T : Inputs)
	{
		inference::ModelInferRequest::InferInputTensor* In = M.add_inputs();
		In->set_name(TCHAR_TO_UTF8(*T.Name));
		In->set_datatype(TCHAR_TO_UTF8(*T.Datatype));
		for (int64 D : T.Shape)
			In->add_shape(D);
		M.add_raw_input_contents(T.Data.GetData(), T.Data.Num());
	}
	for (const FString& N : OutputNames)
		M.add_outputs()->set_name(TCHAR_TO_UTF8(*N));

	++Impl->Pending;
	FScopeLock L(&Impl->Lock);
	Req->Id = Impl->NextId++;
	M.set_id(std::to_string(Req->Id));
	Impl->WriteQueue.Add(MoveTemp(Req));
	Impl->KickWrite();
}

FManipleInferResult FManipleTritonClient::InferSync(
	const FString& Model, TArray<FManipleTensor> Inputs, const TArray<FString>& OutputNames, const FString& Version)
{
	TSharedRef<FManipleInferResult> Result = MakeShared<FManipleInferResult>();
	TSharedRef<bool> Done = MakeShared<bool>(false);
	Infer(Model, MoveTemp(Inputs),
		FManipleInferComplete::CreateLambda(
			[Result, Done](const FManipleInferResult& R)
			{
				*Result = R;
				*Done = true;
			}),
		OutputNames, Version);
	const double Deadline = FPlatformTime::Seconds() + TimeoutSec + 1.0;
	while (!*Done && FPlatformTime::Seconds() < Deadline)
	{
		PumpCompletions();
		FPlatformProcess::Sleep(0.0002f);
	}
	if (!*Done)
		Result->Error = TEXT("InferSync timed out");
	return *Result;
}
