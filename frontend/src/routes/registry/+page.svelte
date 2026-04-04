<script>
    import { onMount } from 'svelte';
    import { api, activeModel, notify } from '$lib/store.js';
    import Spinner from '$lib/components/Spinner.svelte';

    let functions = [];
    let loading = false;

    // Execute panel
    let execFuncId = '';
    let execInputs = '';
    let execResult = null;
    let execError = '';
    let execLoading = false;

    // Eval (by name)
    let evalName = '';
    let evalInputs = '';
    let evalResult = null;
    let evalError = '';
    let evalLoading = false;

    // Register
    let regName = '';
    let regArity = 1;
    let regCompositionRaw = '';
    let regError = '';
    let regLoading = false;

    async function load() {
        loading = true;
        try {
            const data = await api(`/registry/${$activeModel}`);
            functions = data.functions;
        } catch (e) {
            notify(e.message, 'error');
        } finally {
            loading = false;
        }
    }

    onMount(load);
    $effect(() => { $activeModel; load(); });

    async function doExecute() {
        execError = ''; execResult = null; execLoading = true;
        try {
            const inputs = execInputs.split(',').map(s => parseInt(s.trim()));
            const data = await api(`/registry/${$activeModel}/execute`, {
                method: 'POST',
                body: JSON.stringify({ func_id: parseInt(execFuncId), inputs }),
            });
            execResult = data.result;
            notify(`Result: ${data.result}`, 'success');
        } catch (e) {
            execError = e.message;
        } finally {
            execLoading = false;
        }
    }

    async function doEval() {
        evalError = ''; evalResult = null; evalLoading = true;
        try {
            const inputs = evalInputs.split(',').map(s => parseInt(s.trim()));
            const data = await api(`/registry/${$activeModel}/eval`, {
                method: 'POST',
                body: JSON.stringify({ func_name: evalName, inputs }),
            });
            evalResult = data.result;
            notify(`${evalName}(${inputs}) = ${data.result}`, 'success');
        } catch (e) {
            evalError = e.message;
        } finally {
            evalLoading = false;
        }
    }

    async function doRegister() {
        regError = ''; regLoading = true;
        try {
            const composition = JSON.parse(regCompositionRaw);
            const data = await api(`/registry/${$activeModel}/register`, {
                method: 'POST',
                body: JSON.stringify({ name: regName, arity: regArity, composition }),
            });
            notify(`Registered '${regName}' as id=${data.func_id}`, 'success');
            regName = ''; regCompositionRaw = '';
            await load();
        } catch (e) {
            regError = e.message;
        } finally {
            regLoading = false;
        }
    }

    function layerTag(layer) {
        if (layer === 0) return 'tag-blue';
        if (layer === 1) return 'tag-yellow';
        return 'tag-green';
    }
</script>

<div class="page-header">
    <h1>Registry</h1>
    <button class="btn-ghost" onclick={load} disabled={loading}>
        {#if loading}<Spinner />{:else}Refresh{/if}
    </button>
</div>

<div class="layout">
    <section class="panel table-panel">
        <h2>Functions ({functions.length})</h2>
        {#if functions.length === 0}
            <p class="empty">No functions</p>
        {:else}
            <table>
                <thead><tr>
                    <th>ID</th><th>Name</th><th>Arity</th><th>Layer</th>
                </tr></thead>
                <tbody>
                    {#each functions as f}
                        <tr>
                            <td><code>{f.id}</code></td>
                            <td>{f.name}</td>
                            <td>{f.arity}</td>
                            <td><span class="tag {layerTag(f.layer)}">{f.layer === 0 ? 'primitive' : f.layer === 1 ? 'learned' : 'composed'}</span></td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </section>

    <div class="side">
        <div class="panel">
            <h2>Execute by ID</h2>
            <div class="field">
                <label for="exec-fid">Function ID</label>
                <input id="exec-fid" bind:value={execFuncId} placeholder="e.g. 2" type="number" />
            </div>
            <div class="field">
                <label for="exec-in">Inputs (comma-separated)</label>
                <input id="exec-in" bind:value={execInputs} placeholder="e.g. 3, 4" />
            </div>
            {#if execError}<p class="error-msg">{execError}</p>{/if}
            {#if execResult !== null}<p class="success-msg">Result: <code>{execResult}</code></p>{/if}
            <button class="btn-primary" onclick={doExecute} disabled={execLoading || !execFuncId}>
                {#if execLoading}<Spinner />{:else}Execute{/if}
            </button>
        </div>

        <div class="panel">
            <h2>Evaluate by Name</h2>
            <div class="field">
                <label for="eval-name">Function name</label>
                <input id="eval-name" bind:value={evalName} placeholder="e.g. add" />
            </div>
            <div class="field">
                <label for="eval-in">Inputs (comma-separated)</label>
                <input id="eval-in" bind:value={evalInputs} placeholder="e.g. 5, 3" />
            </div>
            {#if evalError}<p class="error-msg">{evalError}</p>{/if}
            {#if evalResult !== null}<p class="success-msg">Result: <code>{evalResult}</code></p>{/if}
            <button class="btn-primary" onclick={doEval} disabled={evalLoading || !evalName}>
                {#if evalLoading}<Spinner />{:else}Evaluate{/if}
            </button>
        </div>

        <div class="panel">
            <h2>Register Composed Function</h2>
            <div class="field">
                <label>Name</label>
                <input bind:value={regName} placeholder="e.g. triple_add" />
            </div>
            <div class="field">
                <label>Arity</label>
                <input type="number" bind:value={regArity} min="1" />
            </div>
            <div class="field">
                <label>Composition JSON <small>(array of [func_id, [arg_indices]])</small></label>
                <textarea rows="4" bind:value={regCompositionRaw} placeholder='[[2,[0,1]],[3,[2]]]'></textarea>
            </div>
            {#if regError}<p class="error-msg">{regError}</p>{/if}
            <button class="btn-primary" onclick={doRegister} disabled={regLoading || !regName}>
                {#if regLoading}<Spinner />{:else}Register{/if}
            </button>
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
    .panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .side { display: flex; flex-direction: column; gap: 1rem; }
    textarea { resize: vertical; font-family: monospace; }
</style>
