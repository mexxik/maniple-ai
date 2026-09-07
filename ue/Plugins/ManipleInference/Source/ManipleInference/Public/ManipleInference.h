#pragma once

#include "Modules/ModuleManager.h"
#include "Logging/LogMacros.h"

MANIPLEINFERENCE_API DECLARE_LOG_CATEGORY_EXTERN(LogManipleInference, Log, All);

class FManipleInferenceModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
