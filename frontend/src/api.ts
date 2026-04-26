import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

const stringifyDetailItem = (item: unknown) => {
  if (typeof item === 'string') return item
  if (!item || typeof item !== 'object') return String(item)

  const detail = item as Record<string, unknown>
  const location = Array.isArray(detail.loc) ? detail.loc.join('.') : ''
  const message = typeof detail.msg === 'string' ? detail.msg : JSON.stringify(detail)
  return location ? `${location}: ${message}` : message
}

export const formatApiError = (error: unknown, fallback: string) => {
  const maybeError = error as {
    response?: { data?: { detail?: unknown; message?: unknown } }
    message?: unknown
  }
  const detail = maybeError?.response?.data?.detail ?? maybeError?.response?.data?.message
  if (Array.isArray(detail)) {
    return detail.map(stringifyDetailItem).join('; ')
  }
  if (typeof detail === 'string') {
    return detail
  }
  if (detail && typeof detail === 'object') {
    return stringifyDetailItem(detail)
  }
  if (typeof maybeError?.message === 'string') {
    return maybeError.message
  }
  return fallback
}
