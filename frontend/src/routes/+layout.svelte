<script>
    import Navbar from '$lib/components/Navbar.svelte';
    import Toast from '$lib/components/Toast.svelte';
    import '$lib/styles/global.css';

    let { children } = $props();
</script>

<svelte:head>
    <title>NSSR</title>
</svelte:head>

<Navbar />
<main>
    {@render children()}
</main>
<Toast />

<style>
    main {
        padding: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
