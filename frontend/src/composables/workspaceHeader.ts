import { reactive } from 'vue'

export interface WorkspaceHeaderMetric {
  label: string
  value: string
}

export const workspaceHeaderState = reactive({
  metrics: [] as WorkspaceHeaderMetric[],
})

export const setWorkspaceHeaderMetrics = (metrics: WorkspaceHeaderMetric[]) => {
  workspaceHeaderState.metrics = metrics
}

export const clearWorkspaceHeaderMetrics = () => {
  workspaceHeaderState.metrics = []
}
