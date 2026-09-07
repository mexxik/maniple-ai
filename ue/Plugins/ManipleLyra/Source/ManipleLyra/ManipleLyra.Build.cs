using UnrealBuildTool;

public class ManipleLyra : ModuleRules
{
	public ManipleLyra(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
		PrivateDependencyModuleNames.AddRange(new string[] {
			"AIModule", "GameplayAbilities", "GameplayTags", "GameplayTasks",
			"ModularGameplay", "ModularGameplayActors",
			"LyraGame",
			"ManipleInference"
		});
	}
}
