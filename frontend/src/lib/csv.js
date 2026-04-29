// Tiny CSV parser/writer. Handles quoted fields, commas, and newlines.

export function parseCSV(text) {
	const rows = [];
	let row = [];
	let cell = '';
	let i = 0;
	let inQuotes = false;
	const src = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

	while (i < src.length) {
		const c = src[i];
		if (inQuotes) {
			if (c === '"') {
				if (src[i + 1] === '"') {
					cell += '"';
					i += 2;
					continue;
				}
				inQuotes = false;
				i++;
				continue;
			}
			cell += c;
			i++;
			continue;
		}
		if (c === '"') {
			inQuotes = true;
			i++;
			continue;
		}
		if (c === ',') {
			row.push(cell);
			cell = '';
			i++;
			continue;
		}
		if (c === '\n') {
			row.push(cell);
			rows.push(row);
			row = [];
			cell = '';
			i++;
			continue;
		}
		cell += c;
		i++;
	}
	if (cell.length > 0 || row.length > 0) {
		row.push(cell);
		rows.push(row);
	}
	return rows.filter((r) => !(r.length === 1 && r[0] === ''));
}

export function stringifyCSV(rows) {
	return rows
		.map((row) =>
			row
				.map((cell) => {
					const s = cell === null || cell === undefined ? '' : String(cell);
					if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
					return s;
				})
				.join(',')
		)
		.join('\n');
}

function normalizeCell(cell) {
	if (cell === null || cell === undefined) return '';
	return String(cell).trim();
}

// Parse an [input_0..input_n, output] table into examples [[inputs[], output], ...]
export function rowsToExamples(rows) {
	if (rows.length === 0) return { examples: [], header: [], error: 'Empty CSV' };
	const header = rows[0];
	const outputIdx = header.length - 1;
	if (outputIdx < 1) return { examples: [], header, error: 'Need at least 1 input + 1 output column' };

	const examples = [];
	for (let r = 1; r < rows.length; r++) {
		const row = rows[r] ?? [];
		const padded = row.slice();
		while (padded.length < header.length) padded.push('');

		const allBlank = padded.every((c) => normalizeCell(c) === '');
		if (allBlank) continue;

		const inputs = [];
		for (let c = 0; c < outputIdx; c++) {
			const raw = normalizeCell(padded[c]);
			if (raw === '') {
				return {
					examples: [],
					header,
					error: `Row ${r + 1}: missing value in input_${c}`
				};
			}
			const n = Number(raw);
			if (Number.isNaN(n)) {
				return {
					examples: [],
					header,
					error: `Row ${r + 1}: invalid number in input_${c} (${raw})`
				};
			}
			inputs.push(n);
		}

		const outputRaw = normalizeCell(padded[outputIdx]);
		if (outputRaw === '') {
			return {
				examples: [],
				header,
				error: `Row ${r + 1}: missing output value`
			};
		}
		const output = Number(outputRaw);
		if (Number.isNaN(output)) {
			return {
				examples: [],
				header,
				error: `Row ${r + 1}: invalid number in output (${outputRaw})`
			};
		}

		examples.push([inputs, output]);
	}

	if (examples.length === 0) {
		return { examples: [], header, error: 'No valid data rows found' };
	}

	return { examples, header };
}

// Kept for backwards compatibility with CSV import path.
export function csvToExamples(rows) {
	return rowsToExamples(rows);
}

export function examplesToCSV(examples) {
	if (!examples || examples.length === 0) return '';
	const maxInputs = Math.max(...examples.map(([inp]) => inp.length));
	const header = [];
	for (let i = 0; i < maxInputs; i++) header.push(`input_${i}`);
	header.push('output');
	const rows = [header];
	for (const [inp, out] of examples) {
		const row = [...inp];
		while (row.length < maxInputs) row.push('');
		row.push(out);
		rows.push(row);
	}
	return stringifyCSV(rows);
}
