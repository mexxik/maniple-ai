using UnrealBuildTool;

public class ManipleLyra : ModuleRules
{
	public ManipleLyra(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
		PrivateDependencyModuleNames.AddRange(new string[] {
			"AIModule", "NavigationSystem", "GameplayAbilities", "GameplayTags", "GameplayTasks",
			"ModularGameplay", "ModularGameplayActors", "GameplayMessageRuntime",
			"LyraGame", "Json",
			"ManipleInference"
		});
	}
}
