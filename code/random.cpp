#include "random.h"

int randomInt(const int range) {
	return randomInt(0, range);
}

int randomInt(const int min, const int range) {
	return rand() % range + min;
}

double randomDouble() {
	return ((double)rand()) / RAND_MAX;
}
