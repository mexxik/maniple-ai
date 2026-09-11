#pragma once

#include "CoreMinimal.h"

/**
 * Observation / action schema v1 for the Lyra bot policy. Sent to the trainer as the AgentSpec (obs dim 32, continuous action dim 5).
 * All spatial values are in the agent's own frame (yaw of its control rotation), Unreal units scaled.
 */
namespace ManipleLyra
{
	constexpr int32 ObsDim = 32;
	constexpr int32 ActDim = 5;
	constexpr int32 NumEnemies = 3; // nearest enemies encoded
	constexpr int32 EnemyStride = 6; // rel x, rel y, rel z, dist, visible, health
	constexpr int32 NumRays = 8; // horizontal wall sensors at eye height, every 45 deg starting straight ahead
	constexpr float PosScale = 2000.f; // cm
	constexpr float VelScale = 600.f; // cm/s
	constexpr float RayRange = 1500.f; // cm, ray value = hit distance / RayRange (1 = nothing within range)

	// observation layout
	constexpr int32 ObsVel = 0; // 3: local velocity / VelScale
	constexpr int32 ObsHealth = 3; // 1: health fraction
	constexpr int32 ObsEnemies = 4; // NumEnemies * EnemyStride = 18
	constexpr int32 ObsPitch = 22; // 1: control pitch / 90
	constexpr int32 ObsBias = 23; // 1: constant 1
	constexpr int32 ObsRays = 24; // NumRays: wall distance per direction

	// action layout
	constexpr int32 ActMoveFwd = 0; // [-1,1]
	constexpr int32 ActMoveRight = 1; // [-1,1]
	constexpr int32 ActYawRate = 2; // [-1,1] * MaxYawDegPerSec
	constexpr int32 ActPitchRate = 3; // [-1,1] * MaxPitchDegPerSec
	constexpr int32 ActFire = 4; // > 0 = hold fire

	constexpr float MaxYawDegPerSec = 120.f;
	constexpr float MaxPitchDegPerSec = 60.f;

	// recording extras: pose of the agent at the observation (world x, y, z in cm, view yaw and pitch in degrees)
	constexpr int32 PoseDim = 5;

	// reward shaping v1 (per decision step); damage is scaled by max health so one full kill's worth of damage = 1
	constexpr float RewardKill = 1.f;
	constexpr float RewardDeath = -1.f;
	constexpr float RewardTeamKill = -1.f;
	constexpr float RewardDamageDealtScale = 1.f; // * damage / max health
	constexpr float RewardDamageTakenScale = -0.5f; // * damage / max health
	constexpr float RewardStep = -0.002f; // small living cost, pushes towards doing something
	constexpr float RewardAimScale = 0.01f; // * (0.5 linear + 0.5 sharp), angle to the nearest visible enemy
	constexpr float AimRangeDeg = 90.f; // linear part: max(0, 1 - angle / AimRangeDeg), a gradient everywhere in front
	constexpr float AimPeakDeg = 6.f; // sharp part: exp(-(angle / AimPeakDeg)^2), pays for precision
	constexpr float RewardShotOnTarget = 0.03f; // a shot fired with a visible enemy within OnTargetDeg (halved while moving)
	constexpr float OnTargetDeg = 5.f;
	constexpr float RewardShotCost = -0.005f; // every real shot: ammo is finite, spraying must not be free
	constexpr int32 SpareAmmoPerEpisode = 60; // spare rounds topped up at every episode start (a pistol carries 15 + 45)
	constexpr float RewardApproachScale = 0.5f; // * (distance closed to the nearest enemy / PosScale), curriculum stages 0-1 only
	constexpr float ApproachStopDist = 1000.f; // cm; inside this range moving is not rewarded (standing still tightens weapon spread)
	constexpr float EpisodeTimeLimitSec = 60.f; // game time; a bot that never dies still closes its trajectory

	static_assert(ObsEnemies + NumEnemies * EnemyStride == ObsPitch, "enemy block must end at ObsPitch");
	static_assert(ObsRays + NumRays == ObsDim, "obs layout size mismatch");
}
