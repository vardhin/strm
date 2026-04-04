import { writable, derived } from 'svelte/store';

export const API_BASE = 'http://localhost:8000';

// Active model name shared across all pages
export const activeModel = writable('default');

// All loaded models (fetched on demand)
export const models = writable({});

// All datasets
export const datasets = writable({});

// Experiment log
export const experiments = writable([]);

// Global notification/toast
export const toast = writable(null);

export function notify(message, type = 'info') {
    toast.set({ message, type });
    setTimeout(() => toast.set(null), 3500);
}

export async function api(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
    }
    return res.json();
}

export async function refreshModels() {
    const data = await api('/models');
    models.set(data);
}

export async function refreshDatasets() {
    const data = await api('/datasets');
    datasets.set(data);
}

export async function refreshExperiments() {
    const data = await api('/experiments');
    experiments.set(data.experiments);
}
