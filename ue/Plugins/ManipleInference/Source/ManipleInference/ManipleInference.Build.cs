using UnrealBuildTool;

public class ManipleInference : ModuleRules
{
	public ManipleInference(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
		PrivateDependencyModuleNames.AddRange(new string[] { "HTTP", "Json", "Projects" });
	}
}
