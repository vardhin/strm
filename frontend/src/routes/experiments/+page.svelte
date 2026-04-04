<script>
    import { onMount } from 'svelte';
    import { experiments, refreshExperiments, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let loading = false;

    async function load() {
        loading = true;
        try { await refreshExperiments(); }
        catch (e) { notify(e.message, 'error'); }
        finally { loading = false; }
    }

    onMount(load);

    let byModel = $derived(groupBy($experiments));

    function groupBy(exps) {
        const m = {};
        for (const e of exps) {
            const k = e.model_name ?? 'unknown';
            if (!m[k]) m[k] = [];
            m[k].push(e);
        }
        return m;
    }

    function successRate(exps) {
        const done = exps.filter(e => e.success !== undefined);
        if (!done.length) return '—';
        const ok = done.filter(e => e.success).length;
        return `${ok}/${done.length}`;
    }
</script>

<div class="page-header">
    <h1>Experiments</h1>
    <button class="btn-ghost" onclick={load} disabled={loading}>
        {#if loading}<Spinner />{:else}Refresh{/if}
    </button>
</div>

{#if $experiments.length === 0}
    <p class="empty" style="margin-top:3rem">
        No experiments recorded yet. Train a model to see results here.
    </p>
{:else}
    <div class="summary-row">
        <div class="stat-card">
            <div class="stat-num">{$experiments.length}</div>
            <div class="stat-label">Total runs</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{$experiments.filter(e => e.success).length}</div>
            <div class="stat-label">Converged</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{Object.keys(byModel).length}</div>
            <div class="stat-label">Models used</div>
        </div>
    </div>

    {#each Object.entries(byModel) as [modelName, exps]}
        <div class="panel" style="margin-top:1rem">
            <div class="model-header">
                <h2>{modelName}</h2>
                <span class="tag tag-blue">success: {successRate(exps)}</span>
            </div>
            <table>
                <thead><tr>
                    <th>#</th><th>Target</th><th>Dataset</th><th>Result</th><th>Elapsed</th><th>Vocab</th><th>Timestamp</th>
                </tr></thead>
                <tbody>
                    {#each exps as exp, i}
                        <tr>
                            <td style="color:#475569">{i + 1}</td>
                            <td><strong>{exp.target}</strong></td>
                            <td>{exp.dataset ?? '—'}</td>
                            <td>
                                {#if exp.success === true}
                                    <span class="tag tag-green">ok</span>
                                {:else if exp.success === false}
                                    <span class="tag tag-red">fail</span>
                                {:else}
                                    <span class="tag tag-blue">—</span>
                                {/if}
                            </td>
                            <td>{exp.elapsed_s ?? '—'}s</td>
                            <td>{exp.vocab_size ?? '—'}</td>
                            <td style="color:#475569;font-size:0.75rem">{exp.timestamp ?? '—'}</td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {/each}
{/if}

<style>
    .summary-row {
        display: flex;
        gap: 1rem;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 100px;
    }
    .stat-num   { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
    .stat-label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.1rem; }

    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
    }
    .model-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
</style>
