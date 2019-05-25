#include "bandit.h"

Bandit::Bandit(const unsigned int numberOfArms,
	const unsigned int rewardBufferSize) : playIndex(0), numberOfArms(numberOfArms), rewardBufferSize(rewardBufferSize)
{
	for (int index = 0; index < numberOfArms; index++) 
	{
		arms.push_back(new Arm(rewardBufferSize));
		actions.push_back(index);
	}
}

Bandit::~Bandit() 
{
	for (int index = 0; index < arms.size(); index++) 
	{
		delete arms[index];
	}
}

int Bandit::play() 
{
	actionPlayCandidates.clear();
	for (int index = 0; index < actions.size(); index++)
	{
		if (getArm(index)->size() > 0)
		{
			actionPlayCandidates.push_back(index);
		}
	}
	return play(actionPlayCandidates);
}

int Bandit::play(const std::vector<int>& legalArms)
{
	means.clear();
	for (int index = 0; index < legalArms.size(); index++)
	{
		means.push_back(arms[legalArms[index]]->mean());
	}
	return legalArms[argmax(means)];
}

void Bandit::update(const double reward) 
{
	if (playIndex >= 0) 
	{
		arms[playIndex]->update(reward);
	}
}

int Bandit::sample() 
{
	playIndex = sampleArm();
	return playIndex;
}

int Bandit::sampleFrom(const std::vector<int>& legalArms)
{
	playIndex = sampleArmFrom(legalArms);
	return playIndex;
}

int EpsilonGreedy::sampleArmFrom(const std::vector<int>& legalArms)
{
	int index = 0;
	if (randomDouble() <= epsilon)
	{
		index = randomInt(legalArms.size());
		return legalArms[index];
	}
	return play(legalArms);
}

UCB1::UCB1(
	const unsigned int numberOfArms,
	const unsigned int rewardBufferSize,
	double explorationConstant) : Bandit(numberOfArms, rewardBufferSize), explorationConstant(explorationConstant) {}

int UCB1::sampleArmFrom(const std::vector<int>& legalArms)
{
	upperConfidences.clear();
	const int numberOfArms = legalArms.size();
	int totalCount = 0;
	for (int index = 0; index < numberOfArms; index++) 
	{
		Arm* arm = getArm(legalArms[index]);
		totalCount += arm->size();
	}
	for (int index = 0; index < numberOfArms; index++)
	{
		Arm* arm = getArm(legalArms[index]);
		const double meanReward = arm->mean();
		const int numberOfRewards = arm->size();
		if (numberOfRewards == 0)
		{
			upperConfidences.push_back(std::numeric_limits<double>::infinity());
		}
		else
		{
			const double explorationTerm = sqrt(2 * log(totalCount) / numberOfArms);
			upperConfidences.push_back(meanReward + explorationConstant*explorationTerm);
		}
	}
	return legalArms[argmax(upperConfidences)];
}

ThompsonSampling::ThompsonSampling(
	const unsigned int numberOfArms,
	const unsigned int rewardBufferSize,
    	const unsigned int updateDelay,
	const unsigned int beta0) : Bandit(numberOfArms, rewardBufferSize), lambda(0), rewardBufferSize(rewardBufferSize), updateDelay(updateDelay), beta0(beta0)
{
	for (int index = 0; index < numberOfArms; index++)
	{
		means.push_back(0);
		vars.push_back(0);
		counts.push_back(0);
	}
}

void ThompsonSampling::update(const double reward)
{
        Bandit::update(reward);
	int currentIndex = currentPlayIndex();
	Arm* arm = getArm(currentIndex);
	int count = arm->size();
	if (count % updateDelay == 0)
	{
		counts[currentIndex] += 1;
		means[currentIndex] = arm->mean();
		double std = arm->std();
		vars[currentIndex] = std*std;
	}
}

void ThompsonSampling::flush()
{

}

void ThompsonSampling::reset()
{
	Bandit::reset();
	int numberOfArms = getNumberOfArms();
	for (int index = 0; index < numberOfArms; index++)
	{
		means[index] = 0;
		vars[index] = 0;
		counts[index] = 0;
	}
}

int ThompsonSampling::sampleArmFrom(const std::vector<int>& legalArms)
{
	sampledMeans.clear();
	int numberOfArms = legalArms.size();
	int rewardBufferSize = getRewardBufferSize();
	for (int index = 0; index < numberOfArms; index++)
	{
		int armIndex = legalArms[index];
		const double mean = means[armIndex];
		const double var = vars[armIndex];
		double n = counts[armIndex];
		if (n == 0)
		{
			sampledMeans.push_back(std::numeric_limits<double>::infinity());
		}
		else
		{
			const double delta = mean - mu0;
			const double lambda1 = lambda0 + n;
			assert(lambda1 > 0);
			const double mu1 = (lambda0*mu0 + n*mean) / lambda1;
			const double alpha1 = alpha0 + n / 2;
			assert(alpha1 >= 1);
			const double beta1 = beta0 + 0.5*(n*var + (lambda0*n*delta*delta) / lambda1);
			assert(beta1 >= 0);
			boost::gamma_distribution<> gd(alpha1, 1/beta1);
			boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > var_gamma(generator, gd);
			double gammaVariate = var_gamma();
			const double normalizedVariance = 1.0 / (lambda1*gammaVariate);
			const double normalizedMean = mu1;
			boost::normal_distribution<double> nd(normalizedMean, sqrt(normalizedVariance));
			boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > var_normal(generator, nd);
			double sampledValue = var_normal();
			sampledMeans.push_back(sampledValue);
		}
	}
	return legalArms[argmax(sampledMeans)];
}
