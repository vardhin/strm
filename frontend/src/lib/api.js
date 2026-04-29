const BASE = 'http://localhost:8000';

async function request(method, path, body) {
	const opts = { method, headers: {} };
	if (body !== undefined) {
		opts.headers['Content-Type'] = 'application/json';
		opts.body = JSON.stringify(body);
	}
	const r = await fetch(`${BASE}${path}`, opts);
	if (!r.ok) {
		const txt = await r.text();
		throw new Error(`${r.status}: ${txt}`);
	}
	return r.json();
}

async function requestEventStream(path, body, onEvent) {
	const r = await fetch(`${BASE}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});

	if (!r.ok) {
		const txt = await r.text();
		throw new Error(`${r.status}: ${txt}`);
	}
	if (!r.body) {
		throw new Error('Streaming response body is not available');
	}

	const reader = r.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let eventName = 'message';
	let dataLines = [];

	function flushEvent() {
		if (dataLines.length === 0) return;
		const raw = dataLines.join('\n');
		let data;
		try {
			data = JSON.parse(raw);
		} catch {
			data = { text: raw };
		}
		onEvent({ event: eventName, data });
		eventName = 'message';
		dataLines = [];
	}

	while (true) {
		const { value, done } = await reader.read();
		if (done) break;

		buffer += decoder.decode(value, { stream: true });

		let idx;
		while ((idx = buffer.indexOf('\n')) !== -1) {
			let line = buffer.slice(0, idx);
			buffer = buffer.slice(idx + 1);

			if (line.endsWith('\r')) line = line.slice(0, -1);
			if (line === '') {
				flushEvent();
				continue;
			}
			if (line.startsWith(':')) continue;

			if (line.startsWith('event:')) {
				eventName = line.slice(6).trim();
			} else if (line.startsWith('data:')) {
				dataLines.push(line.slice(5).trimStart());
			}
		}
	}

	buffer += decoder.decode();
	if (buffer.length > 0) {
		let line = buffer;
		if (line.endsWith('\r')) line = line.slice(0, -1);
		if (line.startsWith('event:')) eventName = line.slice(6).trim();
		else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
	}
	flushEvent();
}

export const api = {
	health: () => request('GET', '/health'),

	listModels: () => request('GET', '/models'),
	getModel: (name) => request('GET', `/models/${name}`),
	saveModel: (name) => request('POST', `/models/${name}/save`),

	listFunctions: (model = 'default') => request('GET', `/registry/${model}`),
	executeFunction: (model, funcId, inputs) =>
		request('POST', `/registry/${model}/execute`, { func_id: funcId, inputs }),
	evalExpression: (model, funcName, inputs) =>
		request('POST', `/registry/${model}/eval`, { func_name: funcName, inputs }),
	registerFunction: (model, name, arity, composition) =>
		request('POST', `/registry/${model}/register`, { name, arity, composition }),

	listDatasets: () => request('GET', '/datasets'),
	getDataset: (name) => request('GET', `/datasets/${name}`),
	createDataset: (name, description, examples) =>
		request('POST', '/datasets', { name, description, examples }),
	deleteDataset: (name) => request('DELETE', `/datasets/${name}`),
	exportCsvUrl: (name) => `${BASE}/datasets/${name}/csv`,

	train: (req) => request('POST', '/train', req),
	trainStream: (req, onEvent) => requestEventStream('/train/stream', req, onEvent),
	runExperiment: (req) => request('POST', '/train/experiment', req),

	testEval: (modelName, datasetName) =>
		request('POST', '/test/eval', { model_name: modelName, dataset_name: datasetName }),
	testPredict: (modelName, inputs) =>
		request('POST', `/test/predict?model_name=${encodeURIComponent(modelName)}`, inputs)
};

export const BASE_URL = BASE;
