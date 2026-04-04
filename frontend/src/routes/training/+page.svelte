<script>
    import { onMount } from 'svelte';
    import { api, activeModel, datasets, refreshDatasets, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    // Single-task train
    let trainDataset = '';
    let trainTarget = '';
    let trainMaxSteps = 10;
    let trainMaxDepth = 3;
    let trainEpochs = 30;
    let trainLoading = false;
    let trainResult = null;
    let trainError = '';

    // Experiment
    let expPreTrain = true;
    let expPreTrainEpochs = 20;
    let expCurriculumRaw = '';
    let expLoading = false;
    let expResult = null;
    let expError = '';

    onMount(async () => {
        await refreshDatasets();
    });

    let datasetNames = $derived(Object.keys($datasets));

    async function doTrain() {
        trainError = ''; trainResult = null; trainLoading = true;
        try {
            trainResult = await api('/train', {
                method: 'POST',
                body: JSON.stringify({
                    model_name: $activeModel,
                    dataset_name: trainDataset,
                    target_name: trainTarget,
                    max_search_steps: trainMaxSteps,
                    max_depth: trainMaxDepth,
                    num_epochs: trainEpochs,
                }),
            });
            notify(trainResult.success ? 'Training succeeded!' : 'Training done (not converged)', trainResult.success ? 'success' : 'info');
        } catch (e) {
            trainError = e.message;
            notify(e.message, 'error');
        } finally {
            trainLoading = false;
        }
    }

    async function doExperiment() {
        expError = ''; expResult = null; expLoading = true;
        try {
            const curriculum = JSON.parse(expCurriculumRaw);
            expResult = await api('/train/experiment', {
                method: 'POST',
                body: JSON.stringify({
                    model_name: $activeModel,
                    curriculum,
                    pre_train: expPreTrain,
                    pre_train_epochs: expPreTrainEpochs,
                }),
            });
            notify('Experiment complete', 'success');
        } catch (e) {
            expError = e.message;
            notify(e.message, 'error');
        } finally {
            expLoading = false;
        }
    }
</script>

<div class="page-header">
    <h1>Training</h1>
    <span class="model-badge">Model: <code>{$activeModel}</code></span>
</div>

<div class="layout">
    <div class="panel">
        <h2>Train on Single Dataset</h2>

        <div class="field">
            <label>Dataset</label>
            {#if datasetNames.length > 0}
                <select bind:value={trainDataset}>
                    <option value="">— select —</option>
                    {#each datasetNames as n}<option value={n}>{n}</option>{/each}
                </select>
            {:else}
                <input bind:value={trainDataset} placeholder="Dataset name" />
            {/if}
        </div>

        <div class="field">
            <label>Target function name</label>
            <input bind:value={trainTarget} placeholder="e.g. add" />
        </div>

        <div class="grid-3">
            <div class="field">
                <label>Max search steps</label>
                <input type="number" bind:value={trainMaxSteps} min="1" />
            </div>
            <div class="field">
                <label>Max depth</label>
                <input type="number" bind:value={trainMaxDepth} min="1" />
            </div>
            <div class="field">
                <label>Epochs</label>
                <input type="number" bind:value={trainEpochs} min="1" />
            </div>
        </div>

        {#if trainError}<p class="error-msg">{trainError}</p>{/if}

        <button class="btn-primary" onclick={doTrain}
            disabled={trainLoading || !trainDataset || !trainTarget}>
            {#if trainLoading}<Spinner /> Training…{:else}Start Training{/if}
        </button>

        {#if trainResult}
            <div class="result-box" class:ok={trainResult.success} class:fail={!trainResult.success}>
                <div class="result-row">
                    <span>Status</span>
                    <span class="tag {trainResult.success ? 'tag-green' : 'tag-red'}">
                        {trainResult.success ? 'Converged' : 'Not converged'}
                    </span>
                </div>
                <div class="result-row">
                    <span>Elapsed</span><span>{trainResult.elapsed_s}s</span>
                </div>
                <div class="result-row">
                    <span>Vocab size</span><span>{trainResult.vocab_size}</span>
                </div>
                <div class="result-row">
                    <span>Timestamp</span><span>{trainResult.timestamp}</span>
                </div>
            </div>
        {/if}
    </div>

    <div class="panel">
        <h2>Run Full Experiment</h2>

        <div class="field">
            <label>
                <input type="checkbox" bind:checked={expPreTrain} style="width:auto;margin-right:0.4rem" />
                Pre-train on curriculum tasks
            </label>
        </div>

        {#if expPreTrain}
            <div class="field">
                <label>Pre-train epochs</label>
                <input type="number" bind:value={expPreTrainEpochs} min="1" />
            </div>
        {/if}

        <div class="field">
            <label>Curriculum JSON <small>(array of CurriculumItem)</small></label>
            <textarea rows="8" bind:value={expCurriculumRaw}
                placeholder={`[
  {
    "dataset_name": "add_pairs",
    "target_name": "add",
    "max_depth": 3,
    "num_epochs": 30
  }
]`}></textarea>
        </div>

        {#if expError}<p class="error-msg">{expError}</p>{/if}

        <button class="btn-primary" onclick={doExperiment}
            disabled={expLoading || !expCurriculumRaw.trim()}>
            {#if expLoading}<Spinner /> Running…{:else}Run Experiment{/if}
        </button>

        {#if expResult}
            <div class="results-list">
                {#each expResult.results as r}
                    <div class="result-row" style="padding:0.4rem 0; border-bottom:1px solid #1e293b;">
                        <span><strong>{r.target ?? r.phase}</strong></span>
                        {#if r.status === 'done'}
                            <span class="tag tag-green">done</span>
                        {:else if r.success === true}
                            <span class="tag tag-green">ok</span>
                        {:else if r.success === false}
                            <span class="tag tag-red">fail</span>
                        {:else if r.status === 'error'}
                            <span class="tag tag-red">error</span>
                        {/if}
                    </div>
                {/each}
            </div>
        {/if}
    </div>
</div>

<style>
    .layout {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        align-items: start;
    }
    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
    }
    .model-badge { font-size: 0.8rem; color: #64748b; }
    textarea { resize: vertical; font-family: monospace; }
    .result-box {
        border-radius: 6px;
        padding: 0.75rem;
        border: 1px solid;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }
    .result-box.ok   { background: #0f2d1a; border-color: #166534; }
    .result-box.fail { background: #2d0f0f; border-color: #7f1d1d; }
    .result-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .results-list { display: flex; flex-direction: column; gap: 0; }
</style>
