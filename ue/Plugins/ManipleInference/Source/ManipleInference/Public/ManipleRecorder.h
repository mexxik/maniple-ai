#pragma once

#include "CoreMinimal.h"
#include "ManipleInferenceTypes.h"

struct FManipleTransitionBatch;
class FJsonObject;
class IFileHandle;

/**
 * Recordings of play: one directory per run, one .npy file per column, rows appended while the game runs.
 * The Python twin (reader, writer, tools) is triton/common/maniple/recording.py; the format is described there.
 *
 *   <dir>/meta.json        producer info + column layout, rewritten on every flush (rows) and at close
 *   <dir>/<column>.npy     [rows, ...], a fixed 128-byte header whose row count is patched on every flush
 */
namespace ManipleNpy
{
	constexpr int32 HeaderSize = 128;

	/** Triton datatype ("FP32", "INT64", "UINT8", "BOOL", "INT32") -> numpy descr ("<f4", ...); empty if unsupported. */
	MANIPLEINFERENCE_API FString Descr(const FString& Datatype);
	MANIPLEINFERENCE_API FString Datatype(const FString& Descr);

	/** The padded version 1.0 header for [Rows, ...RowShape]. */
	MANIPLEINFERENCE_API TArray<uint8> Header(const FString& Descr, int64 Rows, TConstArrayView<int64> RowShape);

	/** Reads a header: returns its total length (bytes before the data), or -1 if the bytes are not a .npy header. */
	MANIPLEINFERENCE_API int32 ParseHeader(TConstArrayView<uint8> Bytes, FString& OutDescr, TArray<int64>& OutShape);
} // namespace ManipleNpy

/**
 * Writes transition batches (the rows a game sends with observe, plus any extra per-row columns such as time,
 * kind or pose) to a recording directory. Columns are fixed by the first Write. Rows are buffered and flushed
 * every FlushRows rows or FlushSeconds seconds, so a crash loses at most that much.
 */
class MANIPLEINFERENCE_API FManipleRecorder
{
public:
	FManipleRecorder() = default;
	~FManipleRecorder();

	/** Creates Dir and writes meta.json from Meta (the recorder adds format, columns, rows, closed). */
	bool Open(const FString& InDir, TSharedPtr<FJsonObject> InMeta);
	bool IsOpen() const { return !Dir.IsEmpty(); }
	const FString& GetDir() const { return Dir; }
	int64 NumRows() const { return Rows + PendingRows; }

	/** Batch rows (inputs, action, reward, done, logp, policy_version, agent_id, episode_id) + Extra columns [Rows, ...]. */
	void Write(const FManipleTransitionBatch& Batch, TConstArrayView<FManipleTensor> Extra = {});
	void Flush();
	void Close();

	int32 FlushRows = 1024;
	double FlushSeconds = 5.0;

	/** <Base>/<Name>-<yyyymmdd-hhmmss>, the default place for a new recording. */
	static FString DefaultDir(const FString& Base, const FString& Name);

private:
	struct FColumn
	{
		FString Name;
		FString Descr;
		TArray<int64> RowShape;
		int64 RowBytes = 0;
		TUniquePtr<IFileHandle> File;
		TArray<uint8> Pending;
	};

	void Append(const FString& Name, const FString& Datatype, TConstArrayView<int64> RowShape, TConstArrayView<uint8> Data, int64 NumRows);
	void WriteMeta(bool bClosed);

	FString Dir;
	TSharedPtr<FJsonObject> Meta;
	TArray<FColumn> Columns;
	bool bLayoutFixed = false;
	bool bFailed = false;
	int64 Rows = 0;
	int64 PendingRows = 0;
	double LastFlush = 0.0;
};

/** A whole recording read into memory: every column as one tensor [rows, ...]. Rows come from the file sizes. */
struct MANIPLEINFERENCE_API FManipleRecording
{
	TSharedPtr<FJsonObject> Meta;
	TMap<FString, FManipleTensor> Columns;
	int64 Rows = 0;

	bool Load(const FString& Dir, FString& OutError);
	const FManipleTensor* Find(const FString& Name) const { return Columns.Find(Name); }
};
