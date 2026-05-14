function createModelStore() {
	let name = $state('default');
	return {
		get name() {
			return name;
		},
		set: (v) => {
			name = v;
			if (typeof localStorage !== 'undefined') localStorage.setItem('modelName', v);
		},
		load: () => {
			if (typeof localStorage !== 'undefined') {
				const v = localStorage.getItem('modelName');
				if (v) name = v;
			}
		}
	};
}

export const modelStore = createModelStore();
