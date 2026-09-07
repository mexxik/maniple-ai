#include "ManipleBatchInferer.h"
#include "ManipleTritonClient.h"
#include "ManipleInference.h"

FManipleBatchInferer::FManipleBatchInferer(TSharedPtr<FManipleTritonClient> InClient, const FString& InModel, const FString& InInputName,
	int32 InInDim, const FString& InOutputName, int32 InOutDim)
	: Client(InClient)
	, Model(InModel)
	, InputName(InInputName)
	, OutputName(InOutputName)
	, InDim(InInDim)
	, OutDim(InOutDim)
{
}

void FManipleBatchInferer::Submit(TConstArrayView<float> Row, FRowCallback Callback)
{
	checkf(Row.Num() == InDim, TEXT("row has %d values, expected %d"), Row.Num(), InDim);
	Rows.Append(Row.GetData(), Row.Num());
	Callbacks.Add(MoveTemp(Callback));
}

bool FManipleBatchInferer::Flush(FBatchCallback OnBatchDone)
{
	const int32 N = Callbacks.Num();
	if (N == 0 || !Client.IsValid())
		return false;

	const int64 Shape[2] = {N, InDim};
	FManipleTensor Input = FManipleTensor::MakeFloat(InputName, Shape, Rows);
	TSharedRef<TArray<FRowCallback>> Pending = MakeShared<TArray<FRowCallback>>(MoveTemp(Callbacks));
	Rows.Reset();
	Callbacks.Reset();

	const FString Out = OutputName;
	const int32 Dim = OutDim;
	Client->Infer(Model, {MoveTemp(Input)},
		FManipleInferComplete::CreateLambda(
			[Pending, Out, Dim, N, OnBatchDone](const FManipleInferResult& R)
			{
				const FManipleTensor* T = R.bSuccess ? R.FindOutput(Out) : nullptr;
				const bool bOk = T && T->AsFloats().Num() == N * Dim;
				if (!bOk && R.bSuccess)
				{
					UE_LOG(LogManipleInference, Warning, TEXT("batch: output '%s' has %d floats, expected %d x %d"), *Out,
						T ? T->AsFloats().Num() : 0, N, Dim);
				}
				for (int32 i = 0; i < N; ++i)
				{
					(*Pending)[i](bOk, bOk ? T->AsFloats().Slice(i * Dim, Dim) : TConstArrayView<float>());
				}
				if (OnBatchDone)
					OnBatchDone(R, N);
			}),
		{Out});
	return true;
}
