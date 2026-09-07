#include "ManipleBotConfig.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"

FManipleBotConfig FManipleBotConfig::FromCommandLine()
{
	FManipleBotConfig C;
	const TCHAR* Cmd = FCommandLine::Get();
	FString Brain;
	if (FParse::Value(Cmd, TEXT("ManipleBrain="), Brain))
	{
		if (Brain.Equals(TEXT("random"), ESearchCase::IgnoreCase)) C.Brain = EManipleBrain::Random;
		else if (Brain.Equals(TEXT("triton"), ESearchCase::IgnoreCase)) C.Brain = EManipleBrain::Triton;
		else C.Brain = EManipleBrain::None;
	}
	FParse::Value(Cmd, TEXT("ManipleModel="), C.Model);
	FParse::Value(Cmd, TEXT("ManipleTritonUrl="), C.TritonUrl);
	FString Bots;
	if (FParse::Value(Cmd, TEXT("ManipleBots="), Bots) && !Bots.Equals(TEXT("all"), ESearchCase::IgnoreCase))
	{
		C.MaxBots = FCString::Atoi(*Bots);
	}
	FParse::Value(Cmd, TEXT("ManipleHz="), C.DecisionHz);
	C.DecisionHz = FMath::Clamp(C.DecisionHz, 1.f, 60.f);
	return C;
}

FString FManipleBotConfig::ToString() const
{
	const TCHAR* BrainStr = Brain == EManipleBrain::Random ? TEXT("random") : Brain == EManipleBrain::Triton ? TEXT("triton") : TEXT("none");
	return FString::Printf(TEXT("brain=%s model=%s url=%s bots=%s hz=%.0f"), BrainStr, *Model, *TritonUrl,
		MaxBots < 0 ? TEXT("all") : *FString::FromInt(MaxBots), DecisionHz);
}
