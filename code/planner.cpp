#include "planner.h"

int POSTS::SelectAction()
{
	int banditIndex = currentIndex%Params.MaxDepth;
	reset();
    Rollout();
	int action = GreedyUCB(Root, false);
	return action;
}

void POSTS::Rollout()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	std::vector<int> legal;
	assert(BeliefState().GetNumSamples() > 0);
	for (int i = 0; i < Params.NumSimulations; i++)
	{
                STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.GenerateActionSpace(*state, GetHistory(), legal, GetStatus(), false);
		
		int banditIndex = currentIndex%Params.MaxDepth;
		int action = bandits[banditIndex]->sampleFrom(legal);
                Simulator.Validate(*state);

		int observation;
		double immediateReward, delayedReward, totalReward;
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);

		delayedReward = Rollout(*state, legal, 1,i);
		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
		Root->Child(action).Value.Add(totalReward);
		bandits[banditIndex]->update(totalReward);
		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

double POSTS::Rollout(
	STATE& state,
	std::vector<int>& legalActions,
	const int t,
	const int i)
{
	if (t >= Params.MaxDepth)
	{
		return 0;
	}
	int numberOfActions = Simulator.GetNumActions();
	bool terminal = false;
	int observation;
	double immediateReward, delayedReward;
	int banditIndex = (currentIndex + t)%Params.MaxDepth;
	if (!terminal) {
		Simulator.GenerateActionSpace(state, GetHistory(), legalActions, GetStatus(), true);
		int action = bandits[banditIndex]->sampleFrom(legalActions);
		terminal = Simulator.Step(state, action, observation, immediateReward);
		History.Add(action, observation);
	}
	if (terminal)
	{
	        bandits[banditIndex]->update(immediateReward);
	        return immediateReward;
	}
	double successorReturn = Rollout(state, legalActions, t+1,i);
	double discount = Simulator.GetDiscount();
	double returnValue = immediateReward + discount*successorReturn;
	bandits[banditIndex]->update(returnValue);
	return returnValue;
}

void POOLTS::TreeSearch()
{
	int historyDepth = History.Size();

	for (int n = 0; n < Params.NumSimulations; n++)
	{
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.Validate(*state);
		Status.Phase = SIMULATOR::STATUS::TREE;
		TreeDepth = 0;
		PeakTreeDepth = 0;
		double totalReward = Simulate(*state, rootNode, 0);
		StatTotalReward.Add(totalReward);
		StatTreeDepth.Add(PeakTreeDepth);
		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

double POOLTS::Simulate(STATE& state, POOLTSNode* node, int t)
{
    std::vector<int> legal;
    Simulator.GenerateActionSpace(state, GetHistory(), legal, GetStatus(), false);
    int action = node->getBandit()->sampleFrom(legal);
    PeakTreeDepth = TreeDepth;
    if (t >= Params.MaxDepth)
    {
    	return 0;
    }
    bool isLeaf = node->IsLeaf();
    if(isLeaf)
    {
        node->Expand();
    }
    int observation;
    double immediateReward, delayedReward = 0;
    bool terminal = Simulator.Step(state, action, observation, immediateReward);
    if(t == 0)
    {
        VNODE*& vnode = Root->Child(action).Child(observation);
        if (!vnode && !terminal) {
            vnode = ExpandNode(&state);
            AddSample(vnode, state);
	}
    }
    History.Add(action, observation);
    if(terminal)
    {
        node->Update(immediateReward);
        return immediateReward;
    }
    assert(observation >= 0 && observation < Simulator.GetNumObservations());

    TreeDepth++;
    delayedReward = isLeaf? Rollout(state) : Simulate(state, node->getNext(action, pool), t+1);
    TreeDepth--;

    double totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
    node->Update(totalReward);
    return totalReward;
}

int SYMBOL::SelectAction()
{
	reset();
	Rollout();
	return GreedyUCB(Root, false);
}

void SYMBOL::Rollout()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	std::vector<int> legal;
	assert(BeliefState().GetNumSamples() > 0);
	for (int i = 0; i < Params.NumSimulations; i++)
	{
		STATE* state = Root->Beliefs().CreateSample(Simulator);
        STATE* firstState = state;
		Simulator.GenerateActionSpace(*state, GetHistory(), legal, GetStatus(), false);
		
		int action = bandits[0]->sampleFrom(legal);
        int firstAction = action;
		Simulator.Validate(*state);

		int observation;
		double immediateReward;
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);
        rewards[0] = immediateReward;
        int stepCount = 1;
        for(int t = 1; t < Params.MaxDepth; t++)
        {
            if(!terminal)
            {
                Simulator.GenerateActionSpace(*state, GetHistory(), legal, GetStatus(), true);
                int action = bandits[t]->sampleFrom(legal);
                terminal = Simulator.Step(*state, action, observation, immediateReward);
                History.Add(action, observation);
                rewards[stepCount] = immediateReward;
                stepCount += 1;
            }
        }
        double returnValue = 0;
        for(int t = stepCount - 1; t >= 0; t--)
        {
            returnValue = rewards[t] + Simulator.GetDiscount()*returnValue;
            rewards[t] = returnValue;
        }
		Root->Child(firstAction).Value.Add(rewards[0]);
        bandits[0]->update(rewards[0]);
        bool predecessorConverged = bandits[0]->hasConverged(banditConvergenceEpsilon);
        int numberOfBandits = 1;
        for(int t = 1; t < stepCount; t++)
        {
            if(predecessorConverged)
            {
                bandits[t]->update(rewards[t]);
                numberOfBandits += 1;
                predecessorConverged = bandits[t]->hasConverged(banditConvergenceEpsilon);
            }
        }
		maxNumberOfBandits = std::max(maxNumberOfBandits, numberOfBandits);
		Simulator.FreeState(firstState);
		History.Truncate(historyDepth);
	}
}
