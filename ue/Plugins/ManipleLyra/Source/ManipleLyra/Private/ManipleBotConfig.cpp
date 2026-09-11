#include "ManipleBotConfig.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/DateTime.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"

const TCHAR* ManipleAgentKindName(EManipleAgentKind Kind)
{
	switch (Kind)
	{
	case EManipleAgentKind::Policy:
		return TEXT("policy");
	case EManipleAgentKind::Heuristic:
		return TEXT("heuristic");
	case EManipleAgentKind::Lyra:
		return TEXT("lyra");
	case EManipleAgentKind::Human:
		return TEXT("human");
	case EManipleAgentKind::Replay:
		return TEXT("replay");
	case EManipleAgentKind::Random:
		return TEXT("random");
	}
	return TEXT("?");
}

FManipleBotConfig FManipleBotConfig::FromCommandLine()
{
	FManipleBotConfig C;
	const TCHAR* Cmd = FCommandLine::Get();

	// ---- brain / mode ----
	FString Brain;
	if (FParse::Value(Cmd, TEXT("ManipleBrain="), Brain))
	{
		if (Brain.Equals(TEXT("random"), ESearchCase::IgnoreCase))
			C.Brain = EManipleBrain::Random;
		else if (Brain.Equals(TEXT("heuristic"), ESearchCase::IgnoreCase))
			C.Brain = EManipleBrain::Heuristic;
		else if (Brain.Equals(TEXT("triton"), ESearchCase::IgnoreCase))
			C.Brain = EManipleBrain::Triton;
		else if (Brain.Equals(TEXT("replay"), ESearchCase::IgnoreCase))
			C.Brain = EManipleBrain::Replay;
		else
			C.Brain = EManipleBrain::None;
	}
	FString Mode;
	if (FParse::Value(Cmd, TEXT("ManipleMode="), Mode) && Mode.Equals(TEXT("train"), ESearchCase::IgnoreCase))
		C.Mode = EManipleMode::Train;

	FParse::Value(Cmd, TEXT("ManipleModel="), C.Model);
	FParse::Value(Cmd, TEXT("ManipleTritonUrl="), C.TritonUrl);

	// ---- channel / exploration ----
	FParse::Value(Cmd, TEXT("ManipleChannel="), C.Channel);
	C.bExplore = C.Mode == EManipleMode::Train;
	int32 Explore = 0;
	if (FParse::Value(Cmd, TEXT("ManipleExplore="), Explore))
		C.bExplore = Explore != 0;
	if (C.Mode == EManipleMode::Train)
		C.Channel = TEXT("latest");

	// ---- bots ----
	FString Bots;
	if (FParse::Value(Cmd, TEXT("ManipleBots="), Bots) && !Bots.Equals(TEXT("all"), ESearchCase::IgnoreCase))
		C.MaxBots = FCString::Atoi(*Bots);
	C.DecisionHz = C.Mode == EManipleMode::Train ? 15.f : 10.f;
	FParse::Value(Cmd, TEXT("ManipleHz="), C.DecisionHz);
	C.DecisionHz = FMath::Clamp(C.DecisionHz, 1.f, 60.f);
	FParse::Value(Cmd, TEXT("ManipleSpawnBots="), C.SpawnBots);

	// ---- network ----
	FParse::Value(Cmd, TEXT("ManipleNet="), C.NetPreset);
	FString Hidden;
	if (FParse::Value(Cmd, TEXT("ManipleHidden="), Hidden))
	{
		TArray<FString> Parts;
		Hidden.ParseIntoArray(Parts, TEXT(","));
		for (const FString& P : Parts)
			C.Hidden.Add(FCString::Atoi(*P));
	}
	FParse::Value(Cmd, TEXT("ManipleActivation="), C.Activation);
	int32 LayerNorm = 0;
	if (FParse::Value(Cmd, TEXT("ManipleLayerNorm="), LayerNorm))
		C.bLayerNorm = LayerNorm != 0;
	FParse::Value(Cmd, TEXT("ManipleEntropy="), C.EntropyCoef);
	FParse::Value(Cmd, TEXT("ManipleLogStd="), C.LogStdInit);

	// ---- training helpers ----
	FParse::Value(Cmd, TEXT("ManipleTimeScale="), C.TimeScale);
	C.TimeScale = FMath::Clamp(C.TimeScale, 0.1f, 50.f);
	C.bSpectate = FParse::Param(Cmd, TEXT("ManipleSpectate"));

	// ---- curriculum ----
	FString Curriculum = C.Mode == EManipleMode::Train ? TEXT("auto") : TEXT("off");
	FParse::Value(Cmd, TEXT("ManipleCurriculum="), Curriculum);
	if (Curriculum.Equals(TEXT("auto"), ESearchCase::IgnoreCase))
	{
		C.CurriculumStage = 0;
		C.bCurriculumAuto = true;
	}
	else if (Curriculum.IsNumeric())
	{
		C.CurriculumStage = FCString::Atoi(*Curriculum);
	}
	C.bEndless = C.Mode == EManipleMode::Train;
	int32 Endless = 0;
	if (FParse::Value(Cmd, TEXT("ManipleEndless="), Endless))
		C.bEndless = Endless != 0;
	C.bSyncAct = C.Mode == EManipleMode::Train;
	int32 Sync = 0;
	if (FParse::Value(Cmd, TEXT("ManipleSync="), Sync))
		C.bSyncAct = Sync != 0;
	FString Opp;
	if (FParse::Value(Cmd, TEXT("ManipleOpponents="), Opp))
	{
		if (Opp.Equals(TEXT("self"), ESearchCase::IgnoreCase))
			C.Opponents = EManipleOpponents::Self;
		else if (Opp.Equals(TEXT("mixed"), ESearchCase::IgnoreCase))
			C.Opponents = EManipleOpponents::Mixed;
		else if (Opp.Equals(TEXT("heuristic"), ESearchCase::IgnoreCase))
			C.Opponents = EManipleOpponents::Heuristic;
		else if (Opp.Equals(TEXT("lyra"), ESearchCase::IgnoreCase))
			C.Opponents = EManipleOpponents::Lyra;
	}

	// ---- score and evaluation ----
	FString Score;
	if (FParse::Value(Cmd, TEXT("ManipleScore="), Score))
	{
		FString Kind, Tag;
		if (!Score.Split(TEXT(":"), &Kind, &Tag))
			Kind = Score;
		if (Kind.Equals(TEXT("stat"), ESearchCase::IgnoreCase))
			C.ScoreKind = EManipleScoreKind::Stat;
		else if (Kind.Equals(TEXT("team"), ESearchCase::IgnoreCase))
			C.ScoreKind = EManipleScoreKind::Team;
		else
			C.ScoreKind = EManipleScoreKind::Kills;
		C.ScoreTag = Tag;
	}
	FParse::Value(Cmd, TEXT("ManipleEval="), C.EvalSeconds);
	if (C.EvalSeconds > 0.f)
		C.bEndless = true; // the evaluation decides when the match is over
	int32 EvalReport = 0;
	if (FParse::Value(Cmd, TEXT("ManipleEvalReport="), EvalReport))
		C.bEvalReport = EvalReport != 0;

	// ---- recording and replay ----
	if (FParse::Value(Cmd, TEXT("ManipleRecord="), C.RecordDir) || FParse::Param(Cmd, TEXT("ManipleRecord")))
	{
		if (C.RecordDir.IsEmpty() || C.RecordDir == TEXT("1"))
			C.RecordDir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Maniple"), TEXT("recordings"),
				C.Model + TEXT("-") + FDateTime::Now().ToString(TEXT("%Y%m%d-%H%M%S")));
		C.RecordDir = FPaths::ConvertRelativePathToFull(C.RecordDir);
	}
	FString Who;
	if (!FApp::CanEverRender())
		C.RecordKinds &= ~(1 << (uint8)EManipleAgentKind::Human); // headless: the local player is an idle pawn, not a human
	if (FParse::Value(Cmd, TEXT("ManipleRecordWho="), Who) && !Who.Equals(TEXT("all"), ESearchCase::IgnoreCase))
	{
		C.RecordKinds = 0;
		TArray<FString> Parts;
		Who.ParseIntoArray(Parts, TEXT(","));
		for (const FString& P : Parts)
		{
			for (uint8 K = 0; K <= (uint8)EManipleAgentKind::Random; ++K)
			{
				if (P.Equals(ManipleAgentKindName((EManipleAgentKind)K), ESearchCase::IgnoreCase))
					C.RecordKinds |= 1 << K;
			}
		}
	}
	FString Replay;
	if (FParse::Value(Cmd, TEXT("ManipleReplay="), Replay))
	{
		FString AgentStr;
		if (Replay.Split(TEXT(":"), &C.ReplayDir, &AgentStr, ESearchCase::CaseSensitive, ESearchDir::FromEnd) && AgentStr.IsNumeric())
			C.ReplayAgent = FCString::Atoi(*AgentStr);
		else
			C.ReplayDir = Replay;
		C.ReplayDir = FPaths::ConvertRelativePathToFull(C.ReplayDir);
	}

	return C;
}

const TCHAR* FManipleBotConfig::OpponentsToString() const
{
	switch (Opponents)
	{
	case EManipleOpponents::Self:
		return TEXT("self");
	case EManipleOpponents::Mixed:
		return TEXT("mixed");
	case EManipleOpponents::Heuristic:
		return TEXT("heuristic");
	case EManipleOpponents::Lyra:
		return TEXT("lyra");
	default:
		return TEXT("stage");
	}
}

FString FManipleBotConfig::ScoreToString() const
{
	switch (ScoreKind)
	{
	case EManipleScoreKind::Stat:
		return TEXT("stat:") + ScoreTag;
	case EManipleScoreKind::Team:
		return TEXT("team:") + ScoreTag;
	default:
		return TEXT("kills");
	}
}

FString FManipleBotConfig::ToString() const
{
	const TCHAR* BrainStr = Brain == EManipleBrain::Random ? TEXT("random")
		: Brain == EManipleBrain::Heuristic				   ? TEXT("heuristic")
		: Brain == EManipleBrain::Triton				   ? TEXT("triton")
		: Brain == EManipleBrain::Replay				   ? TEXT("replay")
														   : TEXT("none");
	const TCHAR* ModeStr = Mode == EManipleMode::Train ? TEXT("train") : TEXT("infer");

	FString HiddenStr;
	for (int32 W : Hidden)
		HiddenStr += (HiddenStr.IsEmpty() ? TEXT("") : TEXT(",")) + FString::FromInt(W);

	const TCHAR* OppStr = OpponentsToString();
	const FString CurriculumStr = CurriculumStage < 0 ? TEXT("off") : bCurriculumAuto ? TEXT("auto") : FString::FromInt(CurriculumStage);

	return FString::Printf(
		TEXT("brain=%s mode=%s model=%s url=%s channel=%s explore=%d bots=%s hz=%.0f spawn=%d net=%s hidden=[%s] "
			 "activation=%s layernorm=%d entropy=%.3f logstd=%.2f timescale=%.1f spectate=%d curriculum=%s opponents=%s endless=%d sync=%d "
			 "score=%s eval=%.0f eval_report=%d record=%s record_who=0x%02x replay=%s replay_agent=%d"),
		BrainStr, ModeStr, *Model, *TritonUrl, *Channel, bExplore ? 1 : 0, MaxBots < 0 ? TEXT("all") : *FString::FromInt(MaxBots),
		DecisionHz, SpawnBots, *NetPreset, *HiddenStr, *Activation, bLayerNorm ? 1 : 0, EntropyCoef, LogStdInit, TimeScale,
		bSpectate ? 1 : 0, *CurriculumStr, OppStr, bEndless ? 1 : 0, bSyncAct ? 1 : 0, *ScoreToString(), EvalSeconds, bEvalReport ? 1 : 0,
		RecordDir.IsEmpty() ? TEXT("off") : *RecordDir, RecordKinds, ReplayDir.IsEmpty() ? TEXT("off") : *ReplayDir, ReplayAgent);
}
