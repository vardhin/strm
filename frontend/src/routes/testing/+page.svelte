<script>
    import { onMount } from 'svelte';
    import { api, activeModel, datasets, models, refreshDatasets, refreshModels, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let evalDataset = '';
    let evalLoading = false;
    let evalResult = null;
    let evalError = '';

    let compareModels = '';
    let compareDataset = '';
    let compareLoading = false;
    let compareResult = null;
    let compareError = '';

    let predictInputs = '';
    let predictLoading = false;
    let predictResult = null;
    let predictError = '';

    onMount(async () => {
        await Promise.all([refreshDatasets(), refreshModels()]);
    });

    let datasetNames = $derived(Object.keys($datasets));
    let modelNames = $derived(Object.keys($models));

    async function doEval() {
        evalError = ''; evalResult = null; evalLoading = true;
        try {
            evalResult = await api('/test/eval', {
                method: 'POST',
                body: JSON.stringify({ model_name: $activeModel, dataset_name: evalDataset }),
            });
        } catch (e) {
            evalError = e.message;
            notify(e.message, 'error');
        } finally {
            evalLoading = false;
        }
    }

    async function doCompare() {
        compareError = ''; compareResult = null; compareLoading = true;
        try {
            const names = compareModels.split(',').map(s => s.trim()).filter(Boolean);
            compareResult = await api('/test/compare', {
                method: 'POST',
                body: JSON.stringify({ model_names: names, dataset_name: compareDataset }),
            });
        } catch (e) {
            compareError = e.message;
            notify(e.message, 'error');
        } finally {
            compareLoading = false;
        }
    }

    async function doPredict() {
        predictError = ''; predictResult = null; predictLoading = true;
        try {
            const inputs = predictInputs.split(',').map(s => parseInt(s.trim()));
            const params = new URLSearchParams({ model_name: $activeModel });
            predictResult = await api(`/test/predict?${params}`, {
                method: 'POST',
                body: JSON.stringify(inputs),
            });
        } catch (e) {
            predictError = e.message;
            notify(e.message, 'error');
        } finally {
            predictLoading = false;
        }
    }

    function pct(n) { return (n * 100).toFixed(1) + '%'; }
</script>

<div class="page-header">
    <h1>Testing</h1>
    <span class="model-badge">Model: <code>{$activeModel}</code></span>
</div>

<div class="layout">
    <div class="panel">
        <h2>Evaluate Model on Dataset</h2>
        <div class="field">
            <label>Dataset</label>
            {#if datasetNames.length > 0}
                <select bind:value={evalDataset}>
                    <option value="">— select —</option>
                    {#each datasetNames as n}<option value={n}>{n}</option>{/each}
                </select>
            {:else}
                <input bind:value={evalDataset} placeholder="Dataset name" />
            {/if}
        </div>
        {#if evalError}<p class="error-msg">{evalError}</p>{/if}
        <button class="btn-primary" onclick={doEval} disabled={evalLoading || !evalDataset}>
            {#if evalLoading}<Spinner /> Evaluating…{:else}Evaluate{/if}
        </button>

        {#if evalResult}
            <div class="accuracy-display">
                <div class="acc-num">{pct(evalResult.accuracy)}</div>
                <div class="acc-sub">{evalResult.correct}/{evalResult.total} correct</div>
            </div>
            <table>
                <thead><tr><th>Inputs</th><th>Expected</th><th>Correct</th><th>Matched by</th></tr></thead>
                <tbody>
                    {#each evalResult.details as d}
                        <tr>
                            <td><code>{JSON.stringify(d.inputs)}</code></td>
                            <td><code>{d.expected}</code></td>
                            <td>
                                {#if d.correct}
                                    <span class="tag tag-green">✓</span>
                                {:else}
                                    <span class="tag tag-red">✗</span>
                                {/if}
                            </td>
                            <td style="font-size:0.75rem;color:#64748b">
                                {d.matching_functions.map(f => f.name).join(', ') || '—'}
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>

    <div class="right">
        <div class="panel">
            <h2>Compare Models</h2>
            <div class="field">
                <label>Model names (comma-separated)</label>
                {#if modelNames.length > 0}
                    <input bind:value={compareModels} placeholder={modelNames.join(', ')} />
                {:else}
                    <input bind:value={compareModels} placeholder="default, model2" />
                {/if}
            </div>
            <div class="field">
                <label>Dataset</label>
                {#if datasetNames.length > 0}
                    <select bind:value={compareDataset}>
                        <option value="">— select —</option>
                        {#each datasetNames as n}<option value={n}>{n}</option>{/each}
                    </select>
                {:else}
                    <input bind:value={compareDataset} placeholder="Dataset name" />
                {/if}
            </div>
            {#if compareError}<p class="error-msg">{compareError}</p>{/if}
            <button class="btn-primary" onclick={doCompare}
                disabled={compareLoading || !compareModels || !compareDataset}>
                {#if compareLoading}<Spinner />{:else}Compare{/if}
            </button>

            {#if compareResult}
                <table>
                    <thead><tr><th>Model</th><th>Accuracy</th><th>Correct</th><th>Vocab</th></tr></thead>
                    <tbody>
                        {#each Object.entries(compareResult.comparison) as [name, info]}
                            <tr>
                                <td>{name}</td>
                                <td><strong>{pct(info.accuracy)}</strong></td>
                                <td>{info.correct}/{info.total}</td>
                                <td>{info.vocab_size}</td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            {/if}
        </div>

        <div class="panel">
            <h2>Neural Prediction</h2>
            <p style="font-size:0.8rem;color:#64748b">Shows what the TRM model predicts — before symbolic verification.</p>
            <div class="field">
                <label>Inputs (comma-separated)</label>
                <input bind:value={predictInputs} placeholder="e.g. 3, 4" />
            </div>
            {#if predictError}<p class="error-msg">{predictError}</p>{/if}
            <button class="btn-primary" onclick={doPredict} disabled={predictLoading || !predictInputs}>
                {#if predictLoading}<Spinner />{:else}Predict{/if}
            </button>

            {#if predictResult}
                <div class="predict-section">
                    <h2>Primary</h2>
                    {#each predictResult.predictions.primary as p}
                        <div class="pred-row">
                            <span>{p.name}</span>
                            <div class="bar-wrap">
                                <div class="bar" style="width:{Math.abs(p.score) * 20}px"></div>
                            </div>
                            <code>{p.score}</code>
                        </div>
                    {/each}
                </div>
                <div class="predict-section">
                    <h2>Composition type</h2>
                    {#each Object.entries(predictResult.predictions.composition) as [type, prob]}
                        <div class="pred-row">
                            <span>{type}</span>
                            <div class="bar-wrap">
                                <div class="bar" style="width:{prob * 120}px"></div>
                            </div>
                            <code>{(prob * 100).toFixed(1)}%</code>
                        </div>
                    {/each}
                </div>
                <div class="result-row" style="justify-content:space-between;font-size:0.82rem;padding-top:0.5rem">
                    <span>Halt probability</span>
                    <code>{(predictResult.predictions.halt_prob * 100).toFixed(1)}%</code>
                </div>
            {/if}
        </div>
    </div>
</div>

<style>
    .layout {
        display: grid;
        grid-template-columns: 1fr 340px;
        gap: 1rem;
        align-items: start;
    }
    .right { display: flex; flex-direction: column; gap: 1rem; }
    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .model-badge { font-size: 0.8rem; color: #64748b; }

    .accuracy-display { text-align: center; padding: 0.5rem 0; }
    .acc-num { font-size: 2.5rem; font-weight: 700; color: #60a5fa; }
    .acc-sub { font-size: 0.8rem; color: #64748b; }

    .predict-section { display: flex; flex-direction: column; gap: 0.3rem; }
    .pred-row {
        display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem;
    }
    .pred-row span { min-width: 70px; color: #94a3b8; }
    .bar-wrap { flex: 1; }
    .bar { height: 6px; background: #3b82f6; border-radius: 3px; min-width: 2px; max-width: 120px; }
    .result-row { display: flex; }
</style>
