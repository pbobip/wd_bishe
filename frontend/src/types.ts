export type SegmentationMode = 'traditional' | 'dl'
export type InputMode = 'single' | 'batch'

export interface ModelRunner {
  id: number
  slot: string
  display_name: string
  python_path: string
  env_name?: string | null
  weight_path?: string | null
  extra_config: Record<string, unknown>
  is_active: boolean
}

export interface RunRecord {
  id: number
  name: string
  input_mode: InputMode
  segmentation_mode: SegmentationMode
  status: string
  progress: number
  error_message?: string | null
  config: Record<string, any>
  summary?: Record<string, any> | null
  chart_data?: Record<string, any> | null
  export_bundle_path?: string | null
  created_at: string
  updated_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface RunStepDetails {
  step_key: string
  status: string
  processed_images: number
  total_images: number
  remaining_images: number
  progress_ratio: number
  current_image_name?: string | null
  message?: string
  updated_at?: string
}

export interface RunResultImage {
  image_id: number
  image_name: string
  input_url: string
  modes: Record<string, {
    summary: Record<string, any>
    artifacts: Record<string, string | null>
  }>
}

export interface RunStep {
  id: number
  step_key: string
  status: string
  message?: string | null
  details?: RunStepDetails | null
  started_at?: string | null
  finished_at?: string | null
}

export interface RunResultsPayload {
  run: RunRecord
  images: RunResultImage[]
  steps: RunStep[]
  exports: Array<{
    id: number
    kind: string
    url: string
    path: string
  }>
}

export interface PostprocessPreviewPayload {
  preview_token: string
  mode: 'traditional' | 'dl'
  image_id: number
  image_name: string
  before_mask_url: string | null
  after_mask_url: string | null
  before_overlay_url: string | null
  after_overlay_url: string | null
  before_object_overlay_url: string | null
  after_object_overlay_url: string | null
  before_summary: Record<string, any>
  after_summary: Record<string, any>
  image_count: number
}

export interface PostprocessConfirmPayload {
  run: RunRecord
  mode: 'traditional' | 'dl'
  image_count: number
}
