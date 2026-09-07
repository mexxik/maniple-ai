#pragma once

#include "CoreMinimal.h"

enum class EManipleBrain : uint8
{
	None,    // Lyra behaviour-tree bots, untouched (default)
	Random,  // take over bots, random actions, no server
	Triton   // take over bots, actions from the Triton model
};

/**
 * Runtime behaviour, all from the command line so nothing changes unless asked:
 *   -ManipleBrain=none|random|triton   -ManipleModel=lyra_policy   -ManipleTritonUrl=http://localhost:8000
 *   -ManipleBots=all|N                 -ManipleHz=10
 *   -ManipleBatch=1|0 (client-side batching: one request per tick for all agents, or one per agent)
 *   -ManipleSpawnBots=N (spawn N extra bots through Lyra's bot creation component, for scaling tests)
 */
struct MANIPLELYRA_API FManipleBotConfig
{
	EManipleBrain Brain = EManipleBrain::None;
	FString Model = TEXT("lyra_policy");
	FString TritonUrl = TEXT("http://localhost:8000");
	int32 MaxBots = -1;        // -1 = all
	float DecisionHz = 10.f;
	bool bBatch = true;
	int32 SpawnBots = 0;

	static FManipleBotConfig FromCommandLine();
	FString ToString() const;
	bool IsActive() const { return Brain != EManipleBrain::None; }
};
