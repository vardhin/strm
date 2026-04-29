<script>
	let { examples = $bindable([]), numInputs = $bindable(2) } = $props();

	function addRow() {
		const inputs = Array.from({ length: numInputs }, () => 0);
		examples = [...examples, [inputs, 0]];
	}

	function deleteRow(i) {
		examples = examples.filter((_, j) => j !== i);
	}

	function updateInput(i, j, value) {
		const n = Number(value);
		if (Number.isNaN(n)) return;
		const next = examples.slice();
		const [inp, out] = next[i];
		const newInp = inp.slice();
		newInp[j] = n;
		next[i] = [newInp, out];
		examples = next;
	}

	function updateOutput(i, value) {
		const n = Number(value);
		if (Number.isNaN(n)) return;
		const next = examples.slice();
		const [inp] = next[i];
		next[i] = [inp, n];
		examples = next;
	}

	function changeNumInputs(delta) {
		const n = Math.max(1, numInputs + delta);
		numInputs = n;
		examples = examples.map(([inp, out]) => {
			const newInp = inp.slice(0, n);
			while (newInp.length < n) newInp.push(0);
			return [newInp, out];
		});
	}
</script>

<div class="editor">
	<div class="row">
		<div class="row-label">
			Inputs per row:
			<span class="mono">{numInputs}</span>
		</div>
		<button onclick={() => changeNumInputs(-1)} disabled={numInputs <= 1}>−</button>
		<button onclick={() => changeNumInputs(1)}>+</button>
		<button onclick={addRow}>+ Add row</button>
		<span class="muted small">{examples.length} rows</span>
	</div>

	<div class="table-scroll">
		<table>
			<thead>
				<tr>
					<th class="idx-col">#</th>
					{#each Array(numInputs) as _, j}
						<th>input_{j}</th>
					{/each}
					<th>output</th>
					<th class="idx-col"></th>
				</tr>
			</thead>
			<tbody>
				{#each examples as [inp, out], i}
					<tr>
						<td class="mono muted">{i}</td>
						{#each Array(numInputs) as _, j}
							<td>
								<input
									type="number"
									step="any"
									value={inp[j] ?? 0}
									oninput={(e) => updateInput(i, j, e.target.value)}
								/>
							</td>
						{/each}
						<td>
							<input
								type="number"
								step="any"
								value={out}
								oninput={(e) => updateOutput(i, e.target.value)}
							/>
						</td>
						<td>
							<button class="danger small-btn" onclick={() => deleteRow(i)} aria-label="delete">
								×
							</button>
						</td>
					</tr>
				{/each}
				{#if examples.length === 0}
					<tr>
						<td colspan={numInputs + 3} class="muted">No rows. Click "+ Add row".</td>
					</tr>
				{/if}
			</tbody>
		</table>
	</div>
</div>

<style>
	.editor {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.table-scroll {
		max-height: 450px;
		overflow: auto;
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	input[type='number'] {
		width: 100%;
		min-width: 70px;
		padding: 0.25rem 0.4rem;
		font-family: var(--mono);
		font-size: 0.85rem;
	}

	.small-btn {
		padding: 0.1rem 0.45rem;
		font-size: 1rem;
		line-height: 1;
	}

	.row-label {
		color: var(--text-dim);
		font-size: 0.9rem;
	}

	.idx-col {
		width: 3rem;
	}
</style>
