#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "ManipleBotConfig.h"
#include "ManipleBotSubsystem.generated.h"

class FManipleTritonClient;
class UManipleAgentComponent;
class AAIController;

/**
 * Created only when -ManipleBrain is set on the command line. Finds Lyra bot controllers
 * (server side), attaches a UManipleAgentComponent to their pawns (also after respawn),
 * owns the shared Triton client, and logs aggregate stats.
 */
UCLASS()
class MANIPLELYRA_API UManipleBotSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual bool ShouldCreateSubsystem(UObject* Outer) const override;
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override { RETURN_QUICK_DECLARE_CYCLE_STAT(UManipleBotSubsystem, STATGROUP_Tickables); }
	virtual bool IsTickable() const override { return Config.IsActive(); }

	const FManipleBotConfig& GetConfig() const { return Config; }

private:
	void ScanForBots();
	void LogStats();

	FManipleBotConfig Config;
	TSharedPtr<FManipleTritonClient> Client;
	/** controllers we own; a controller keeps its brain across pawn respawns */
	TArray<TWeakObjectPtr<AAIController>> OwnedControllers;
	float ScanTimer = 0.f;
	float StatsTimer = 0.f;
	int32 NextAgentId = 0;
};
