#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include "random.h"
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <time.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

class Arm
{
public:
	Arm(const unsigned int capacity) : count(0), capacity(capacity)
	{
		for (int index = 0; index < capacity + 1; index++)
        {
			lastEstimatedValues.push_back(0);
		}
		value = std::numeric_limits<double>::infinity();
		squaredValue = std::numeric_limits<double>::infinity();
	}

	~Arm() {}

	void update(const double reward) 
	{
		if (count == 0)
		{
			value = 0;
			squaredValue = 0;
		}
		count += 1;
		value += reward;
		squaredValue += reward*reward;
	}
    
    const bool hasConverged(const double epsilon)
    {
        if(count < capacity + 1)
        {
            return false;
        }
        double deltaSum = 0.0;
        for(int index = 1; index < capacity + 1; index++)
        {
            deltaSum += std::abs(lastEstimatedValues[index]-lastEstimatedValues[index-1]);
        }
        return deltaSum/capacity < epsilon;
    }

	const double mean() 
	{
	        if(count == 0)
		{
		      return 0;
		}
	        return value/count;
	}

	void setValues(const double newValue, const double newSquaredValue, const int newCount)
	{
		value = newValue;
		squaredValue = newSquaredValue;
		count = newCount;
	}

	const double std() 
	{
		double mean = this->mean();
		if (count == 0) {
			return 0;
		}
		double meanSquared = mean*mean;
		double expectedSquaredSum = squaredValue / count;
		double res = expectedSquaredSum - meanSquared;
		if(res < 0) {
            res = 0;
		}
		res = sqrt(res);
		return res;
	}

	const unsigned int size() const
	{
		return count;
	}

	void reset()
	{
		value = std::numeric_limits<double>::infinity();
		squaredValue = std::numeric_limits<double>::infinity();
		count = 0;
	}
private:
	unsigned int count;
	const unsigned int capacity;
    std::vector<double> lastEstimatedValues;
	double value;
	double squaredValue;
};

class Bandit
{
public:
	Bandit(const unsigned int numberOfArms,
		const unsigned int rewardBufferSize);
	virtual ~Bandit();

	virtual int sampleArm()
	{
		return sampleArmFrom(actions);
	}
	virtual int play();
	int currentPlayIndex() const
	{
		return playIndex;
	}

	virtual int play(const std::vector<int>& legalArms);
	int sample();
	virtual int sampleArmFrom(const std::vector<int>& legalArms) = 0;
	int sampleFrom(const std::vector<int>& legalArms);
	virtual void update(const double reward);
	const unsigned int getNumberOfArms() 
	{
		return numberOfArms;
	}
	Arm* getArm(const int index) 
	{
		return arms[index];
	}
    const bool hasConverged(const double epsilon)
    {
        if (playIndex >= 0) 
        {
            return arms[playIndex]->hasConverged(epsilon);
        }
        return false;
    }
	int argmax(std::vector<double>& data) 
	{
		static std::vector<int> candidateValueIndices;
		candidateValueIndices.clear();
		double bestValue = -std::numeric_limits<double>::infinity();
		int n = data.size();
		for (int index = 0; index < n; index++)
		{
			double value = data[index];
			if (value >= bestValue)
			{
				if (value > bestValue)
				{
					candidateValueIndices.clear();
				}
				bestValue = value;
				candidateValueIndices.push_back(index);
			}
		}
		return candidateValueIndices[randomInt(candidateValueIndices.size())];
	}

	virtual void reset()
	{
		int n = arms.size();
		for (int index = 0; index < n; index++)
		{
			arms[index]->reset();
		}
	}

	const unsigned int getRewardBufferSize() const {
		return rewardBufferSize;
	}

private:
	int playIndex;
	const unsigned int numberOfArms;
	const unsigned int rewardBufferSize;
	std::vector<Arm*> arms;
	std::vector<double> means;
	std::vector<int> actions;
	std::vector<int> actionPlayCandidates;
};

class RandomBandit : public Bandit 
{
public:
	RandomBandit(const unsigned int numberOfArms) : Bandit(numberOfArms,1) {}
	virtual ~RandomBandit() {}
	virtual int sampleArmFrom(const std::vector<int>& legalArms)
	{
		return legalArms[randomInt(legalArms.size())];
	}
};

class EpsilonGreedy : public Bandit 
{
public:
	EpsilonGreedy(
		const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
		double epsilon) : Bandit(numberOfArms, rewardBufferSize), epsilon(epsilon) {}
	virtual ~EpsilonGreedy() {}
	virtual int sampleArmFrom(const std::vector<int>& legalArms);
private:
	const double epsilon;
};

class UCB1 : public Bandit
{
public:
	UCB1(
		const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
		double explorationConstant);
	virtual ~UCB1() {}
	virtual int sampleArmFrom(const std::vector<int>& legalArms);
private:
	const double explorationConstant;
	std::vector<double> upperConfidences;
};

class ThompsonSampling : public Bandit 
{
public:
	ThompsonSampling(
		const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
                const unsigned int updateDelay,
		const unsigned int beta0);
	virtual ~ThompsonSampling() {}
	virtual void update(const double reward);
	virtual void reset();
	virtual int sampleArmFrom(const std::vector<int>& legalArms);
	void flush();

	void setBetaAndLambda(double beta, double lambda) 
	{
		beta0 = beta;
		lambda0 = lambda;
	}
private:
	double mu0 = 0;
	double lambda0 = 0.01;
	double alpha0 = 1;
	double beta0;
	double lambda;
        int updateDelay;
	int rewardBufferSize;
	boost::mt19937 generator;
	std::vector<int> counts;
	std::vector<double> sampledMeans;
	std::vector<double> means;
	std::vector<double> vars;
};
