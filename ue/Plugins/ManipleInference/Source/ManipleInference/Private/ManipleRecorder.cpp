#include "ManipleRecorder.h"
#include "ManipleAgentClient.h"
#include "ManipleInference.h"
#include "Dom/JsonObject.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformTime.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

// ---------- .npy headers ----------

namespace ManipleNpy
{
	static const uint8 Magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};

	FString Descr(const FString& Datatype)
	{
		if (Datatype == TEXT("FP32"))
			return TEXT("<f4");
		if (Datatype == TEXT("INT64"))
			return TEXT("<i8");
		if (Datatype == TEXT("INT32"))
			return TEXT("<i4");
		if (Datatype == TEXT("UINT8"))
			return TEXT("|u1");
		if (Datatype == TEXT("BOOL"))
			return TEXT("|b1");
		return FString();
	}

	FString Datatype(const FString& InDescr)
	{
		if (InDescr == TEXT("<f4"))
			return TEXT("FP32");
		if (InDescr == TEXT("<i8"))
			return TEXT("INT64");
		if (InDescr == TEXT("<i4"))
			return TEXT("INT32");
		if (InDescr == TEXT("|u1"))
			return TEXT("UINT8");
		if (InDescr == TEXT("|b1"))
			return TEXT("BOOL");
		return FString();
	}

	TArray<uint8> Header(const FString& InDescr, int64 Rows, TConstArrayView<int64> RowShape)
	{
		FString Shape = FString::Printf(TEXT("(%lld"), Rows);
		for (int64 D : RowShape)
			Shape += FString::Printf(TEXT(", %lld"), D);
		Shape += RowShape.Num() == 0 ? TEXT(",)") : TEXT(")");
		const FString Text = FString::Printf(TEXT("{'descr': '%s', 'fortran_order': False, 'shape': %s, }"), *InDescr, *Shape);

		TArray<uint8> Out;
		Out.Append(Magic, sizeof(Magic));
		const uint16 Len = HeaderSize - sizeof(Magic) - 2;
		Out.Add((uint8)(Len & 0xff));
		Out.Add((uint8)(Len >> 8));
		const FTCHARToUTF8 Utf8(*Text);
		checkf(Utf8.Length() + 1 <= Len, TEXT("npy header does not fit: %s"), *Text);
		Out.Append((const uint8*)Utf8.Get(), Utf8.Length());
		while (Out.Num() < HeaderSize - 1)
			Out.Add(' ');
		Out.Add('\n');
		return Out;
	}

	int32 ParseHeader(TConstArrayView<uint8> Bytes, FString& OutDescr, TArray<int64>& OutShape)
	{
		if (Bytes.Num() < (int32)sizeof(Magic) + 2 || FMemory::Memcmp(Bytes.GetData(), Magic, sizeof(Magic)) != 0)
			return -1;
		const int32 Len = Bytes[sizeof(Magic)] | (Bytes[sizeof(Magic) + 1] << 8);
		const int32 Total = sizeof(Magic) + 2 + Len;
		if (Bytes.Num() < Total)
			return -1;
		const FString Text = FString::ConstructFromPtrSize((const ANSICHAR*)Bytes.GetData() + sizeof(Magic) + 2, Len);

		int32 At = Text.Find(TEXT("'descr': '"));
		if (At < 0)
			return -1;
		At += 10;
		const int32 End = Text.Find(TEXT("'"), ESearchCase::CaseSensitive, ESearchDir::FromStart, At);
		OutDescr = Text.Mid(At, End - At);

		At = Text.Find(TEXT("'shape': ("));
		if (At < 0)
			return -1;
		At += 10;
		const int32 Close = Text.Find(TEXT(")"), ESearchCase::CaseSensitive, ESearchDir::FromStart, At);
		TArray<FString> Parts;
		Text.Mid(At, Close - At).ParseIntoArray(Parts, TEXT(","));
		OutShape.Reset();
		for (FString& P : Parts)
		{
			P.TrimStartAndEndInline();
			if (!P.IsEmpty())
				OutShape.Add(FCString::Atoi64(*P));
		}
		return Total;
	}
} // namespace ManipleNpy

// ---------- recorder ----------

FManipleRecorder::~FManipleRecorder()
{
	Close();
}

FString FManipleRecorder::DefaultDir(const FString& Base, const FString& Name)
{
	return FPaths::Combine(Base, Name + TEXT("-") + FDateTime::Now().ToString(TEXT("%Y%m%d-%H%M%S")));
}

bool FManipleRecorder::Open(const FString& InDir, TSharedPtr<FJsonObject> InMeta)
{
	Close();
	IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
	if (!PF.CreateDirectoryTree(*InDir))
	{
		UE_LOG(LogManipleInference, Error, TEXT("recorder: cannot create %s"), *InDir);
		return false;
	}
	Dir = InDir;
	Meta = InMeta.IsValid() ? InMeta : MakeShared<FJsonObject>();
	Rows = PendingRows = 0;
	bLayoutFixed = bFailed = false;
	LastFlush = FPlatformTime::Seconds();
	WriteMeta(false);
	UE_LOG(LogManipleInference, Display, TEXT("recording to %s"), *Dir);
	return true;
}

void FManipleRecorder::Append(
	const FString& Name, const FString& Datatype, TConstArrayView<int64> RowShape, TConstArrayView<uint8> Data, int64 NumRows)
{
	if (bFailed)
		return;
	FColumn* Col = Columns.FindByPredicate([&](const FColumn& C) { return C.Name == Name; });
	if (!Col)
	{
		if (bLayoutFixed)
		{
			UE_LOG(LogManipleInference, Error, TEXT("recorder: column '%s' appeared after the first write, recording stopped"), *Name);
			bFailed = true;
			return;
		}
		const FString Descr = ManipleNpy::Descr(Datatype);
		if (Descr.IsEmpty())
		{
			UE_LOG(LogManipleInference, Warning, TEXT("recorder: column '%s' has datatype %s, not recorded"), *Name, *Datatype);
			return;
		}
		int64 RowBytes = FManipleTensor::ElementSize(Datatype);
		for (int64 D : RowShape)
			RowBytes *= D;
		IFileHandle* File =
			FPlatformFileManager::Get().GetPlatformFile().OpenWrite(*FPaths::Combine(Dir, Name + TEXT(".npy")), false, true);
		if (!File)
		{
			UE_LOG(LogManipleInference, Error, TEXT("recorder: cannot open %s/%s.npy, recording stopped"), *Dir, *Name);
			bFailed = true;
			return;
		}
		const TArray<uint8> Head = ManipleNpy::Header(Descr, 0, RowShape);
		File->Write(Head.GetData(), Head.Num());
		Col = &Columns.Emplace_GetRef();
		Col->Name = Name;
		Col->Descr = Descr;
		Col->RowShape = TArray<int64>(RowShape);
		Col->RowBytes = RowBytes;
		Col->File.Reset(File);
	}
	if (Col->RowShape != TArray<int64>(RowShape) || Data.Num() != Col->RowBytes * NumRows)
	{
		UE_LOG(LogManipleInference, Error, TEXT("recorder: column '%s' changed layout (%lld bytes for %lld rows), recording stopped"),
			*Name, (int64)Data.Num(), NumRows);
		bFailed = true;
		return;
	}
	Col->Pending.Append(Data.GetData(), Data.Num());
}

void FManipleRecorder::Write(const FManipleTransitionBatch& Batch, TConstArrayView<FManipleTensor> Extra)
{
	if (!IsOpen() || bFailed || Batch.Num() == 0)
		return;
	const int64 N = Batch.Num();
	auto Bytes = [](const auto& Array)
	{
		return TConstArrayView<uint8>((const uint8*)Array.GetData(), Array.Num() * sizeof(Array[0]));
	};

	for (const FManipleTensor& T : Batch.InputTensors())
		Append(T.Name, T.Datatype, TConstArrayView<int64>(T.Shape).Slice(1, T.Shape.Num() - 1), T.Data, N);
	Append(TEXT("action"), TEXT("FP32"), {Batch.ActDim}, Bytes(Batch.Action), N);
	Append(TEXT("reward"), TEXT("FP32"), {}, Bytes(Batch.Reward), N);
	TArray<uint8> Done;
	Done.Reserve(N);
	for (bool D : Batch.Done)
		Done.Add(D ? 1 : 0);
	Append(TEXT("done"), TEXT("BOOL"), {}, Done, N);
	Append(TEXT("logp"), TEXT("FP32"), {}, Bytes(Batch.LogP), N);
	Append(TEXT("policy_version"), TEXT("INT64"), {}, Bytes(Batch.PolicyVersion), N);
	Append(TEXT("agent_id"), TEXT("INT64"), {}, Bytes(Batch.AgentId), N);
	Append(TEXT("episode_id"), TEXT("INT64"), {}, Bytes(Batch.EpisodeId), N);
	for (const FManipleTensor& T : Extra)
	{
		if (T.Shape.Num() == 0 || T.Shape[0] != N)
		{
			UE_LOG(LogManipleInference, Error, TEXT("recorder: extra column '%s' has %lld rows, batch has %lld; recording stopped"),
				*T.Name, T.Shape.Num() ? T.Shape[0] : 0, N);
			bFailed = true;
			return;
		}
		Append(T.Name, T.Datatype, TConstArrayView<int64>(T.Shape).Slice(1, T.Shape.Num() - 1), T.Data, N);
	}
	bLayoutFixed = true;
	PendingRows += N;
	if (PendingRows >= FlushRows || FPlatformTime::Seconds() - LastFlush > FlushSeconds)
		Flush();
}

void FManipleRecorder::Flush()
{
	if (!IsOpen())
		return;
	if (PendingRows > 0)
	{
		Rows += PendingRows;
		PendingRows = 0;
		for (FColumn& C : Columns)
		{
			C.File->SeekFromEnd(0);
			C.File->Write(C.Pending.GetData(), C.Pending.Num());
			C.Pending.Reset();
			C.File->Seek(0);
			const TArray<uint8> Head = ManipleNpy::Header(C.Descr, Rows, C.RowShape);
			C.File->Write(Head.GetData(), Head.Num());
			C.File->Flush();
		}
		WriteMeta(false);
	}
	LastFlush = FPlatformTime::Seconds();
}

void FManipleRecorder::Close()
{
	if (!IsOpen())
		return;
	Flush();
	for (FColumn& C : Columns)
		C.File.Reset(); // closes
	WriteMeta(true);
	UE_LOG(LogManipleInference, Display, TEXT("recording closed: %lld rows in %s"), Rows, *Dir);
	Columns.Reset();
	Dir.Empty();
}

void FManipleRecorder::WriteMeta(bool bClosed)
{
	TSharedPtr<FJsonObject> Out = MakeShared<FJsonObject>();
	Out->Values = Meta->Values;
	Out->SetStringField(TEXT("format"), TEXT("maniple-recording"));
	Out->SetNumberField(TEXT("format_version"), 1);
	TSharedPtr<FJsonObject> Cols = MakeShared<FJsonObject>();
	for (const FColumn& C : Columns)
	{
		TSharedPtr<FJsonObject> J = MakeShared<FJsonObject>();
		J->SetStringField(TEXT("descr"), C.Descr);
		TArray<TSharedPtr<FJsonValue>> Shape;
		for (int64 D : C.RowShape)
			Shape.Add(MakeShared<FJsonValueNumber>((double)D));
		J->SetArrayField(TEXT("shape"), Shape);
		Cols->SetObjectField(C.Name, J);
	}
	Out->SetObjectField(TEXT("columns"), Cols);
	Out->SetNumberField(TEXT("rows"), (double)Rows);
	Out->SetBoolField(TEXT("closed"), bClosed);

	FString Text;
	const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Text);
	FJsonSerializer::Serialize(Out.ToSharedRef(), Writer);
	const FString Path = FPaths::Combine(Dir, TEXT("meta.json"));
	FFileHelper::SaveStringToFile(Text, *(Path + TEXT(".tmp")));
	IFileManager::Get().Move(*Path, *(Path + TEXT(".tmp")), true, true);
}

// ---------- reader ----------

bool FManipleRecording::Load(const FString& Dir, FString& OutError)
{
	Columns.Reset();
	Rows = 0;
	FString Text;
	if (!FFileHelper::LoadFileToString(Text, *FPaths::Combine(Dir, TEXT("meta.json"))))
	{
		OutError = FString::Printf(TEXT("%s: no meta.json"), *Dir);
		return false;
	}
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Meta) || !Meta.IsValid())
	{
		OutError = FString::Printf(TEXT("%s: meta.json is not valid JSON"), *Dir);
		return false;
	}

	TArray<FString> Files;
	IFileManager::Get().FindFiles(Files, *FPaths::Combine(Dir, TEXT("*.npy")), true, false);
	if (Files.Num() == 0)
	{
		OutError = FString::Printf(TEXT("%s: no columns"), *Dir);
		return false;
	}
	Rows = INT64_MAX;
	for (const FString& File : Files)
	{
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *FPaths::Combine(Dir, File)))
		{
			OutError = FString::Printf(TEXT("%s: cannot read %s"), *Dir, *File);
			return false;
		}
		FString Descr;
		TArray<int64> Shape;
		const int32 Head = ManipleNpy::ParseHeader(Bytes, Descr, Shape);
		const FString Datatype = ManipleNpy::Datatype(Descr);
		if (Head < 0 || Shape.Num() == 0 || Datatype.IsEmpty())
		{
			OutError = FString::Printf(TEXT("%s: %s is not a recording column"), *Dir, *File);
			return false;
		}
		int64 RowBytes = FManipleTensor::ElementSize(Datatype);
		for (int32 i = 1; i < Shape.Num(); ++i)
			RowBytes *= Shape[i];
		const int64 FileRows = RowBytes > 0 ? (Bytes.Num() - Head) / RowBytes : 0;

		FManipleTensor T;
		T.Name = FPaths::GetBaseFilename(File);
		T.Datatype = Datatype;
		T.Shape = Shape;
		T.Shape[0] = FileRows;
		T.Data.Append(Bytes.GetData() + Head, FileRows * RowBytes);
		Columns.Add(T.Name, MoveTemp(T));
		Rows = FMath::Min(Rows, FileRows);
	}
	return true;
}
