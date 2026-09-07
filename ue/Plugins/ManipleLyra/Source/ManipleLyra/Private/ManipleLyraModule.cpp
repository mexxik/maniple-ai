#include "ManipleLyra.h"

DEFINE_LOG_CATEGORY(LogManipleLyra);

void FManipleLyraModule::StartupModule()
{
	UE_LOG(LogManipleLyra, Log, TEXT("ManipleLyra module started"));
}

IMPLEMENT_MODULE(FManipleLyraModule, ManipleLyra)
