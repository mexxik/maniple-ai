#include "ManipleAgentComponent.h"
#include "ManipleLyra.h"
#include "Character/LyraCharacter.h"
#include "Character/LyraHealthComponent.h"
#include "Teams/LyraTeamSubsystem.h"
#include "AbilitySystem/LyraAbilitySystemComponent.h"
#include "AIController.h"
#include "BrainComponent.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "CollisionQueryParams.h"
#include "GameplayTagContainer.h"

using namespace ManipleLyra;

UManipleAgentComponent::UManipleAgentComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;
}

void UManipleAgentComponent::BeginPlay()
{
	Super::BeginPlay();
	Character = Cast<ALyraCharacter>(GetOwner());
	AI = Character.IsValid() ? Cast<AAIController>(Character->GetController()) : nullptr;
	if (!Character.IsValid() || !AI.IsValid())
	{
		UE_LOG(LogManipleLyra, Warning, TEXT("agent %d: owner is not an AI-controlled LyraCharacter, disabling"), AgentId);
		SetComponentTickEnabled(false);
		return;
	}
	TakeOverFromBehaviorTree();
	UE_LOG(LogManipleLyra, Display, TEXT("agent %d: took over %s (%s)"), AgentId, *Character->GetName(), *AI->GetName());
}

void UManipleAgentComponent::TakeOverFromBehaviorTree()
{
	if (UBrainComponent* Brain = AI->FindComponentByClass<UBrainComponent>())
	{
		if (Brain->IsRunning())
			Brain->StopLogic(TEXT("Maniple"));
	}
	AI->StopMovement();
	AI->ClearFocus(EAIFocusPriority::Gameplay);
}

bool UManipleAgentComponent::IsReady() const
{
	if (!Character.IsValid() || !AI.IsValid())
		return false;
	const ULyraHealthComponent* H = ULyraHealthComponent::FindHealthComponent(Character.Get());
	return H && !H->IsDeadOrDying();
}

void UManipleAgentComponent::BuildObservation(TArray<float>& Obs) const
{
	Obs.SetNumZeroed(ObsDim);
	const ALyraCharacter* Me = Character.Get();
	const FRotator Frame(0.f, AI->GetControlRotation().Yaw, 0.f);
	const FVector MyLoc = Me->GetActorLocation();

	const FVector LocalVel = Frame.UnrotateVector(Me->GetVelocity()) / VelScale;
	Obs[ObsVel + 0] = LocalVel.X;
	Obs[ObsVel + 1] = LocalVel.Y;
	Obs[ObsVel + 2] = LocalVel.Z;

	if (const ULyraHealthComponent* H = ULyraHealthComponent::FindHealthComponent(Me))
	{
		Obs[ObsHealth] = H->GetMaxHealth() > 0.f ? H->GetHealth() / H->GetMaxHealth() : 0.f;
	}

	struct FEnemy
	{
		const ALyraCharacter* C;
		float Dist;
		float Health;
	};
	TArray<FEnemy> Enemies;
	const ULyraTeamSubsystem* Teams = GetWorld()->GetSubsystem<ULyraTeamSubsystem>();
	for (TActorIterator<ALyraCharacter> It(GetWorld()); It; ++It)
	{
		const ALyraCharacter* Other = *It;
		if (Other == Me)
			continue;
		const ULyraHealthComponent* OH = ULyraHealthComponent::FindHealthComponent(Other);
		if (!OH || OH->IsDeadOrDying())
			continue;
		if (Teams && Teams->CompareTeams(Me, Other) != ELyraTeamComparison::DifferentTeams)
			continue;
		Enemies.Add({Other, (float)FVector::Dist(MyLoc, Other->GetActorLocation()),
			OH->GetMaxHealth() > 0.f ? OH->GetHealth() / OH->GetMaxHealth() : 0.f});
	}
	Enemies.Sort([](const FEnemy& A, const FEnemy& B) { return A.Dist < B.Dist; });

	FCollisionQueryParams Params(SCENE_QUERY_STAT(ManipleVisibility), false, Me);
	const FVector Eye = Me->GetPawnViewLocation();
	for (int32 i = 0; i < FMath::Min(NumEnemies, Enemies.Num()); ++i)
	{
		const FEnemy& E = Enemies[i];
		const FVector Target = E.C->GetActorLocation();
		const FVector Rel = Frame.UnrotateVector(Target - MyLoc) / PosScale;
		FHitResult Hit;
		const bool bBlocked = GetWorld()->LineTraceSingleByChannel(Hit, Eye, Target, ECC_Visibility, Params) && Hit.GetActor() != E.C;
		float* O = &Obs[ObsEnemies + i * EnemyStride];
		O[0] = Rel.X;
		O[1] = Rel.Y;
		O[2] = Rel.Z;
		O[3] = E.Dist / PosScale;
		O[4] = bBlocked ? 0.f : 1.f;
		O[5] = E.Health;
	}
	Obs[ObsPad + 1] = 1.f; // bias
}

void UManipleAgentComponent::SetAction(TConstArrayView<float> InAction)
{
	if (InAction.Num() < ActDim)
		return;
	for (int32 i = 0; i < ActDim; ++i)
		Action[i] = InAction[i];
	bHasAction = true;
}

void UManipleAgentComponent::SetRandomAction()
{
	Action[ActMoveFwd] = FMath::FRandRange(-1.f, 1.f);
	Action[ActMoveRight] = FMath::FRandRange(-1.f, 1.f);
	Action[ActYawRate] = FMath::FRandRange(-1.f, 1.f);
	Action[ActPitchRate] = FMath::FRandRange(-0.3f, 0.3f);
	Action[ActFire] = FMath::FRand() < 0.3f ? 1.f : -1.f;
	Action[ActJump] = FMath::FRand() < 0.05f ? 1.f : -1.f;
	bHasAction = true;
}

void UManipleAgentComponent::ApplyAction(float DeltaTime)
{
	ALyraCharacter* Me = Character.Get();
	AAIController* Ctrl = AI.Get();

	if (UBrainComponent* Brain = Ctrl->FindComponentByClass<UBrainComponent>())
	{
		if (Brain->IsRunning())
			Brain->StopLogic(TEXT("Maniple"));
	}

	FRotator Rot = Ctrl->GetControlRotation();
	Rot.Yaw += FMath::Clamp(Action[ActYawRate], -1.f, 1.f) * MaxYawDegPerSec * DeltaTime;
	Rot.Pitch = FMath::Clamp(Rot.Pitch + FMath::Clamp(Action[ActPitchRate], -1.f, 1.f) * MaxPitchDegPerSec * DeltaTime, -80.f, 80.f);
	Rot.Roll = 0.f;
	Ctrl->SetControlRotation(Rot);

	const FRotator Frame(0.f, Rot.Yaw, 0.f);
	Me->AddMovementInput(Frame.RotateVector(FVector::ForwardVector), FMath::Clamp(Action[ActMoveFwd], -1.f, 1.f));
	Me->AddMovementInput(Frame.RotateVector(FVector::RightVector), FMath::Clamp(Action[ActMoveRight], -1.f, 1.f));

	if (Action[ActJump] > 0.f && Me->CanJump())
		Me->Jump();

	if (ULyraAbilitySystemComponent* ASC = Me->GetLyraAbilitySystemComponent())
	{
		static const FGameplayTag FireTag = FGameplayTag::RequestGameplayTag(TEXT("InputTag.Weapon.Fire"));
		const bool bWantFire = Action[ActFire] > 0.f;
		if (bWantFire != bFiring)
		{
			bWantFire ? ASC->AbilityInputTagPressed(FireTag) : ASC->AbilityInputTagReleased(FireTag);
			bFiring = bWantFire;
		}
		ASC->ProcessAbilityInput(DeltaTime, false); // bots have no player controller doing this
	}
}

void UManipleAgentComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (!bHasAction || !IsReady())
		return;
	ApplyAction(DeltaTime);
}
