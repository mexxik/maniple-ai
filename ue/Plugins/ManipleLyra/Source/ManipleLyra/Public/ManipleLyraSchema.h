#pragma once

#include "CoreMinimal.h"

/**
 * Observation / action schema v1 for the Lyra bot policy. Must match python/export_policy.py.
 * All spatial values are in the agent's own frame (yaw of its control rotation), Unreal units scaled.
 */
namespace ManipleLyra
{
	constexpr int32 ObsDim = 24;
	constexpr int32 ActDim = 6;
	constexpr int32 NumEnemies = 3; // nearest enemies encoded
	constexpr int32 EnemyStride = 6; // rel x, rel y, rel z, dist, visible, health
	constexpr float PosScale = 2000.f; // cm
	constexpr float VelScale = 600.f; // cm/s

	// observation layout
	constexpr int32 ObsVel = 0; // 3: local velocity / VelScale
	constexpr int32 ObsHealth = 3; // 1: health fraction
	constexpr int32 ObsEnemies = 4; // NumEnemies * EnemyStride = 18
	constexpr int32 ObsPad = 22; // 2: reserved (0, bias 1)

	// action layout
	constexpr int32 ActMoveFwd = 0; // [-1,1]
	constexpr int32 ActMoveRight = 1; // [-1,1]
	constexpr int32 ActYawRate = 2; // [-1,1] * MaxYawDegPerSec
	constexpr int32 ActPitchRate = 3; // [-1,1] * MaxPitchDegPerSec
	constexpr int32 ActFire = 4; // > 0 = hold fire
	constexpr int32 ActJump = 5; // > 0 = jump

	constexpr float MaxYawDegPerSec = 180.f;
	constexpr float MaxPitchDegPerSec = 90.f;

	static_assert(ObsEnemies + NumEnemies * EnemyStride == ObsPad, "enemy block must end at ObsPad");
	static_assert(ObsPad + 2 == ObsDim, "obs layout size mismatch");
}
