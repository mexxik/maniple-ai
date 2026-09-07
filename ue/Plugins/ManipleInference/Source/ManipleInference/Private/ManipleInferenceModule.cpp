#include "ManipleInference.h"

DEFINE_LOG_CATEGORY(LogManipleInference);

void FManipleInferenceModule::StartupModule()
{
	UE_LOG(LogManipleInference, Log, TEXT("ManipleInference module started"));
}

void FManipleInferenceModule::ShutdownModule() {}

IMPLEMENT_MODULE(FManipleInferenceModule, ManipleInference)
