import type { RunRecord } from '../types'
import { api } from '../api'

export interface CalibrationState {
  calibrated: boolean
  value: number | null
  label: string
  detail: string
  hint: string | null
}

export interface CalibrationProbe {
  preview_url?: string | null
  footer_detected: boolean
  background_cropped?: boolean
  scale_bar_detected: boolean
  scale_bar_pixels: number | null
  analysis_width_px: number | null
  source_width_px: number | null
  analysis_height_px: number | null
  ocr_available?: boolean
  ocr_scale_bar_um?: number | null
  ocr_fov_um?: number | null
  ocr_magnification_text?: string | null
  ocr_wd_mm?: number | null
  ocr_detector?: string | null
  ocr_scan_mode?: string | null
  ocr_vacuum_mode?: string | null
  ocr_date_text?: string | null
  ocr_time_text?: string | null
  suggested_um_per_px?: number | null
  common_scale_candidates?: Array<{
    scale_um: number
    um_per_px: number
  }>
  message?: string
}

const toFinitePositiveNumber = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return null
  }
  return value
}

export const getUmPerPx = (run: RunRecord | null | undefined): number | null => {
  const value = run?.config?.input_config?.um_per_px
  return toFinitePositiveNumber(value)
}

export const getCalibrationHint = (run: RunRecord | null | undefined): string | null => {
  const summary = (run?.summary ?? {}) as Record<string, unknown>
  const config = (run?.config ?? {}) as Record<string, unknown>
  const inputConfig = (config.input_config ?? {}) as Record<string, unknown>
  const candidates = [
    summary.calibration_hint,
    summary.calibration_message,
    (summary.calibration as Record<string, unknown> | undefined)?.hint,
    inputConfig.calibration_hint,
    inputConfig.calibration_message,
    config.calibration_hint,
  ]

  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim()
    }
  }
  return null
}

export const getCalibrationState = (run: RunRecord | null | undefined): CalibrationState => {
  const value = getUmPerPx(run)
  const hint = getCalibrationHint(run)
  return {
    calibrated: value !== null,
    value,
    label: value !== null ? `已标定 · ${value.toFixed(4)} um/px` : '未标定 · 当前统计仅为 px / px²',
    detail:
      value !== null
        ? '当前任务会按填写的 um_per_px 换算真实面积、尺寸和通道宽度。'
        : '未填写 um_per_px 时，当前任务只输出像素单位统计；真实物理单位结果不可用。',
    hint,
  }
}

export const buildUmPerPxFromScaleBar = (scaleBarUm: number | null, scaleBarPixels: number | null): number | null => {
  const um = toFinitePositiveNumber(scaleBarUm)
  const pixels = toFinitePositiveNumber(scaleBarPixels)
  if (um === null || pixels === null) {
    return null
  }
  return um / pixels
}

export const inspectCalibrationProbe = async (files: File[]): Promise<CalibrationProbe | null> => {
  if (!files.length) {
    return null
  }

  const payload = new FormData()
  files.slice(0, 1).forEach((file) => payload.append('file', file))

  try {
    const response = await api.post<CalibrationProbe>('/calibration/inspect', payload, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  } catch {
    return null
  }
}
