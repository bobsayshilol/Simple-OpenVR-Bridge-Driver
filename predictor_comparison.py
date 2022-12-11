#!/usr/bin/python
import numpy as np

############################
# Simulated movement classes
############################

# Linear motion with gradient |m| passing through (t0,h0)
class GenerateLinear:
	def __init__(self, t0, h0, m):
		self.t0 = t0
		self.h0 = h0
		self.m = m

	def at(self, t):
		return self.h0 + self.m * (t - self.t0)

	def __str__(self):
		return f"{type(self).__name__}({self.t0}, {self.h0}, {self.m})"

# SHM with scale |scale| at frequency |hz|
class GenerateSHM:
	def __init__(self, scale, hz):
		self.scale = scale
		self.hz = hz

	def at(self, t):
		return self.scale * np.sin(2 * np.pi * self.hz * t)

	def __str__(self):
		return f"{type(self).__name__}({self.scale}, {self.hz})"

# Impulse step from |h0| to |h1| at time |t|
class GenerateStep:
	def __init__(self, h0, t, h1):
		self.t = t
		self.h0 = h0
		self.h1 = h1

	def at(self, t):
		return np.where(t < self.t, self.h0, self.h1)

	def __str__(self):
		return f"{type(self).__name__}({self.h0}, {self.t}, {self.h1})"

# Smoothstep from (t0,h0) to (t1,h1)
class GenerateSmooth:
	def __init__(self, t0, h0, t1, h1):
		self.t0 = t0
		self.h0 = h0
		self.t1 = t1
		self.h1 = h1

	def at(self, t):
		res = np.where(t <= self.t0, self.h0, 0).astype(float)
		res += np.where(t >= self.t1, self.h1, 0)
		s = self.h0 + t * t * (3 - 2 * t) * (self.h1 - self.h0)
		res += np.where((self.t0 < t) & (t < self.t1), s, 0)
		return res

	def __str__(self):
		return f"{type(self).__name__}({self.t0}, {self.h0}, {self.t1}, {self.h1})"


###################
# Predictor classes
###################

# Linear regression model
class PredictorLinear:
	def predict(self, ts, xs, t):
		avg_t = np.mean(ts)
		std_t = np.std(ts)
		avg_x = np.mean(xs)
		txs = ts * xs
		avg_tx = np.mean(txs)
		m = (avg_tx - avg_t * avg_x) / (std_t * std_t)
		return avg_x + m * (t - avg_t)

	def __str__(self):
		return f"{type(self).__name__}()"

# Simple velocity model
class PredictorVelocity:
	def predict(self, ts, xs, t):
		dx = xs[-1] - xs[-2]
		dt = ts[-1] - ts[-2]
		v = dx / dt
		return xs[-1] + v * (t - ts[-1])

	def __str__(self):
		return f"{type(self).__name__}()"

# Weighted velocity model
class PredictorWeightedVelocity:
	def predict(self, ts, xs, t):
		v = 0.0
		ws = 0.0
		for i in range(len(ts) - 1):
			idx = -1 - i
			dx = xs[idx] - xs[idx - 1]
			dt = ts[idx] - ts[idx - 1]
			w = 1.0 / (1.0 + i)
			v += w * dx / dt
			ws += w
		v /= ws
		return xs[-1] + v * (t - ts[-1])

	def __str__(self):
		return f"{type(self).__name__}()"


##############
# Test harness
##############

class Harness:
	def __init__(self, generator, predictor, fps, latency):
		self.generator = generator
		self.predictor = predictor
		self.spf = 1 / fps
		self.rng = np.random.default_rng()
		# Setup 1s of timestamps
		self.ts = np.arange(0, 1, self.spf) - latency

	def perturb(self, data, delta):
		data += (2 * self.rng.random(data.shape) - 1) * delta
		return data

	def test(self, ts, t1):
		ts = self.perturb(ts, self.spf * 0.02) # +-2% of a frame
		xs = self.generator.at(ts)
		xs = self.perturb(xs, 0.01) # +-1cm measurement error
		predicted = self.predictor.predict(ts, xs, t1)
		actual = self.generator.at(t1)
		return predicted - actual

	def run(self, window):
		repetitions = 10
		deltas = []
		for start in range(len(self.ts) - window + 1):
			for _ in range(repetitions):
				win = self.ts[start : start + window]
				t = (start + window) * self.spf
				delta = self.test(win, t)
				deltas.append(delta)
		return np.mean(deltas), np.std(deltas)


####################
# Setup tests to run
####################

predictors = [
	PredictorLinear(),
	PredictorVelocity(),
	PredictorWeightedVelocity(),
]
generators = [
	# Basic movement
	GenerateLinear(0, 0.1, 0.2),
	GenerateLinear(0.2, 0.9, -2),
	# Circular movements
	GenerateSHM(0.8, 0.2),
	GenerateSHM(0.6, 2.1),
	GenerateSHM(0.3, 6),
	# Sudden jumps
	GenerateStep(0.2, 0.5, 0.21), # 1cm
	GenerateStep(0.4, 0.5, 0.3), # 10cm
	# Smooth movement
	GenerateSmooth(0.2, -0.4, 1.1, 0.5), # slow
	GenerateSmooth(0.4, -0.4, 0.6, 0.5), # fast
]
fpss = [
	10,
	20,
	30,
]
latencies = [
	#0,
	0.001,
	0.005,
	0.02,
	0.05,
]
windows = [
	#1,
	2,
	#5,
	10,
	#20,
]


###############
# Run the tests
###############

from collections import namedtuple
Result = namedtuple('Result', 'predictor generator fps latency window mean stddev')
results = []
for predictor in predictors:
	for generator in generators:
		for fps in fpss:
			for latency in latencies:
				for window in windows:
					# We only test 1s of data, so fps must be at least window size
					if fps >= window:
						harness = Harness(generator, predictor, fps, latency)
						mean,stddev = harness.run(window)
						results.append(Result(predictor, generator, fps, latency, window, mean, stddev))

# Dump it all as a CSV
with open('out.csv', 'w+') as f:
	f.write("predictor,generator,fps,latency,window,mean,stddev\n")
	for result in results:
		f.write(f'"{result.predictor}","{result.generator}",{result.fps},{result.latency},{result.window},{result.mean},{result.stddev}\n')


##########################
# Do some basic processing
##########################

predictor_counts = {}
for generator in generators:
	for fps in fpss:
		for latency in latencies:
			# Find best for this combo, over different window sizes
			filtered = []
			for window in windows:
				for result in results:
					if result.fps == fps and result.latency == latency and result.window == window and result.generator == generator:
						filtered.append(result)

			# Sort them
			def scorer(elem):
				# TODO: factor in stddev better
				return abs(elem.mean) + 2 * elem.stddev
			filtered.sort(key=scorer)

			print(f"Top 3 for {generator} / {fps}fps / {latency}s latency:")
			for idx in range(3):
				result = filtered[idx]
				name = f"{result.predictor}"
				print(f"\t{name:30} window={result.window}:\t{result.mean:.5}\t{result.stddev:.5}\t({scorer(result):.5})")

				# Score based on ranking
				if result.predictor not in predictor_counts:
					predictor_counts[result.predictor] = 0
				predictor_counts[result.predictor] += 1 / (1 + idx) # scorer(result)

print("Final scores:")
for predictor, count in predictor_counts.items():
	print(f"\t{predictor}: {count:.1f}")
