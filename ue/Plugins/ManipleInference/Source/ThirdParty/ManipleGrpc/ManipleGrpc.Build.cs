using UnrealBuildTool;
using System.IO;

/**
 * Prebuilt gRPC (+protobuf/abseil/boringssl/re2/c-ares/zlib) bundled into one static archive.
 * Produce it with scripts/build_grpc_linux.sh (Linux) / scripts/build_grpc_win.ps1 (Win64).
 */
public class ManipleGrpc : ModuleRules
{
	public ManipleGrpc(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		string Include = Path.Combine(ModuleDirectory, "include");
		string Lib;
		if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			Lib = Path.Combine(ModuleDirectory, "lib", "Linux", "libmaniple_grpc.a");
			PublicAdditionalLibraries.Add(Path.Combine(ModuleDirectory, "lib", "Linux", "libmaniple_triton.a"));
			PublicSystemLibraries.AddRange(new string[] { "pthread", "dl" });
		}
		else if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			Lib = Path.Combine(ModuleDirectory, "lib", "Win64", "maniple_grpc.lib");
			PublicAdditionalLibraries.Add(Path.Combine(ModuleDirectory, "lib", "Win64", "maniple_triton.lib"));
			PublicSystemLibraries.AddRange(new string[] { "ws2_32.lib", "crypt32.lib", "advapi32.lib" });
		}
		else
		{
			throw new BuildException("ManipleGrpc: platform " + Target.Platform + " not supported");
		}

		if (!File.Exists(Lib) || !Directory.Exists(Include))
		{
			throw new BuildException("ManipleGrpc: prebuilt gRPC not found (" + Lib + "). Run scripts/build_grpc_linux.sh (or the Win64 script) from the maniple-ai repo root.");
		}

		PublicSystemIncludePaths.Add(Include);
		PublicSystemIncludePaths.Add(Path.Combine(ModuleDirectory, "gen"));   // Triton protos, see scripts/gen_triton_protos.sh
		PublicAdditionalLibraries.Add(Lib);
		PublicDefinitions.Add("GOOGLE_PROTOBUF_NO_RTTI=1");
	}
}
