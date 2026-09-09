#pragma once

#include "CoreMinimal.h"

enum class EManipleBrain : uint8
{
	None, // Lyra behaviour-tree bots, untouched (default)
	Random, // take over bots, random actions, no server
	Heuristic, // take over bots, turn to the nearest visible enemy and shoot; no server (sanity baseline)
	Triton // take over bots, actions from a Maniple policy on Triton
};

enum class EManipleMode : uint8
{
	Infer, // act only: -ManipleChannel picks the served version (default best)
	Train // act from the trainer's latest weights with exploration and send transitions back
};

enum class EManipleOpponents : uint8
{
	Stage, // whatever the curriculum stage says (default)
	Self, // every bot runs the policy (self-play)
	Mixed, // every second bot runs the heuristic brain
	Heuristic, // one whole team runs the heuristic brain
	Lyra // one whole team keeps Lyra's behaviour trees (untouched bots): the benchmark opponent
};

/**
 * What "score" means for this game mode. The subsystem turns it into one number per game-minute, reports it to the
 * trainer (train mode, so 'best' ranks by it) and prints it at the end of an evaluation (-ManipleEval).
 * Only bots in the final curriculum stage / Lyra spawns count, so the number always means "in the real game".
 */
enum class EManipleScoreKind : uint8
{
	Kills, // policy kills per agent-minute (from Lyra.Elimination.Message; deathmatch, elimination)
	Stat, // per-player stat tag on the player state, summed over policy bots, per agent-minute (e.g. ShooterGame.Score.ControlPointCapture)
	Team // team tag stack: our team minus the best other team, per game-minute (e.g. ShooterGame.ControlPoint.TeamScore)
};

/**
 * Runtime behaviour, all from the command line so nothing changes unless asked:
 *   -ManipleBrain=none|random|heuristic|triton    -ManipleMode=infer|train      -ManipleModel=lyra
 *   -ManipleTritonUrl=localhost:8001     -ManipleChannel=best|stable|latest|<version>   -ManipleExplore=0|1
 *   -ManipleBots=all|N                   -ManipleHz=10                 -ManipleSpawnBots=N
 *   -ManipleNet=auto|small|medium|large  -ManipleHidden=256,256        -ManipleActivation=tanh|relu|elu|gelu
 *   -ManipleLayerNorm=0|1                -ManipleEntropy=0.01          -ManipleLogStd=-1.0
 *   -ManipleTimeScale=1.0                -ManipleSpectate
 *   -ManipleCurriculum=auto|off|N        -ManipleOpponents=stage|self|mixed|heuristic|lyra
 *   -ManipleEndless=0|1                  no score / time limit, the match never restarts (default 1 in train mode)
 *   -ManipleSync=0|1                     block the game thread until the policy answers, so every action lasts exactly one
 *                                        decision period regardless of wall-clock speed (default 1 in train mode)
 *   -ManipleScore=kills|stat:<Tag>|team:<Tag>   what counts as score in this game mode (see EManipleScoreKind)
 *   -ManipleEval=N                       evaluation: once the bots can take damage, run N game-seconds, print one
 *                                        "eval:" line with the score and quit (any brain / mode)
 *   -ManipleEvalReport=0|1               also send the evaluation score to the trainer for the served version (default 0)
 */
struct MANIPLELYRA_API FManipleBotConfig
{
	EManipleBrain Brain = EManipleBrain::None;
	EManipleMode Mode = EManipleMode::Infer;
	FString Model = TEXT("lyra");
	FString TritonUrl = TEXT("localhost:8001");
	FString Channel = TEXT("best"); // train mode always uses latest
	bool bExplore = false; // train mode default true
	int32 MaxBots = -1; // -1 = all
	float DecisionHz = 10.f;
	int32 SpawnBots = 0;

	// network spec sent with register (train mode)
	FString NetPreset = TEXT("auto");
	TArray<int32> Hidden;
	FString Activation = TEXT("tanh");
	bool bLayerNorm = false;
	float EntropyCoef = 0.01f; // PPO entropy bonus, keeps exploration alive (-ManipleEntropy=)
	float LogStdInit = -1.f; // initial action noise: log std (-1 = std 0.37 of the action range) (-ManipleLogStd=)

	// training helpers
	float TimeScale = 1.f; // world time dilation
	bool bSpectate = false; // local player follows the bots instead of playing

	// curriculum (train mode): -1 = off, otherwise the stage to start at; bCurriculumAuto = advance on kill rate
	int32 CurriculumStage = -1;
	bool bCurriculumAuto = false;
	EManipleOpponents Opponents = EManipleOpponents::Stage;
	bool bEndless = false;
	bool bSyncAct = false; // wait for the act reply inside the tick (deterministic timing; train default 1)

	// score and evaluation
	EManipleScoreKind ScoreKind = EManipleScoreKind::Kills;
	FString ScoreTag; // stat / team tag name
	float EvalSeconds = 0.f; // > 0: evaluation run of this many game-seconds after the warmup
	bool bEvalReport = false;

	FString ScoreToString() const;
	const TCHAR* OpponentsToString() const;

	static FManipleBotConfig FromCommandLine();
	FString ToString() const;
	bool IsActive() const { return Brain != EManipleBrain::None; }
	bool IsTraining() const { return Brain == EManipleBrain::Triton && Mode == EManipleMode::Train; }
};
