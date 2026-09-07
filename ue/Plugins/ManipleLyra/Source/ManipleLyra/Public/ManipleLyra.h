#pragma once

#include "Modules/ModuleManager.h"
#include "Logging/LogMacros.h"

MANIPLELYRA_API DECLARE_LOG_CATEGORY_EXTERN(LogManipleLyra, Log, All);

class FManipleLyraModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
};
