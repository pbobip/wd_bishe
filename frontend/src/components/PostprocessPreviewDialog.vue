<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    modelValue: boolean
    preview: {
      imageName: string
      beforeImageUrl: string | null
      afterImageUrl: string | null
      beforeLabel?: string
      afterLabel?: string
      message?: string
    } | null
    confirmLoading?: boolean
  }>(),
  {
    confirmLoading: false,
  },
)

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  cancel: []
  confirm: []
}>()

const closeDialog = () => {
  emit('update:modelValue', false)
}

const handleCancel = () => {
  emit('cancel')
  closeDialog()
}

const handleConfirm = () => {
  emit('confirm')
}
</script>

<template>
  <el-dialog
    :model-value="props.modelValue"
    width="min(1080px, 92vw)"
    top="6vh"
    destroy-on-close
    append-to-body
    class="postprocess-preview-dialog"
    @close="handleCancel"
  >
    <template #header>
      <div class="dialog-head">
        <div>
          <strong>后处理预览确认</strong>
          <p>{{ props.preview?.imageName ?? '当前图像' }}</p>
        </div>
      </div>
    </template>

    <div class="dialog-body">
      <div class="dialog-notice">
        <strong>当前展示的是选中图像的前后对比。</strong>
        <p>
          {{ props.preview?.message ?? '确认后将对当前结果统一应用后处理，并刷新主界面的已确认结果。' }}
        </p>
      </div>

      <div class="preview-grid">
        <section class="preview-panel">
          <div class="preview-head">
            <span>处理前</span>
            <strong>{{ props.preview?.beforeLabel ?? '当前确认结果' }}</strong>
          </div>
          <div class="preview-media">
            <img
              v-if="props.preview?.beforeImageUrl"
              :src="props.preview.beforeImageUrl"
              :alt="props.preview?.beforeLabel ?? '处理前图像'"
            />
            <el-empty v-else description="暂无处理前预览" />
          </div>
        </section>

        <section class="preview-panel">
          <div class="preview-head">
            <span>处理后</span>
            <strong>{{ props.preview?.afterLabel ?? '后处理预览结果' }}</strong>
          </div>
          <div class="preview-media">
            <img
              v-if="props.preview?.afterImageUrl"
              :src="props.preview.afterImageUrl"
              :alt="props.preview?.afterLabel ?? '处理后图像'"
            />
            <el-empty v-else description="暂无处理后预览" />
          </div>
        </section>
      </div>
    </div>

    <template #footer>
      <div class="dialog-actions">
        <el-button @click="handleCancel">取消</el-button>
        <el-button type="primary" :loading="props.confirmLoading" @click="handleConfirm">确认应用</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped>
.dialog-head strong {
  display: block;
  font-size: 20px;
  line-height: 1.3;
}

.dialog-head p {
  margin: 6px 0 0;
  color: var(--muted);
  line-height: 1.5;
}

.dialog-body {
  display: grid;
  gap: 16px;
}

.dialog-notice {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(23, 96, 135, 0.08);
  border: 1px solid rgba(23, 96, 135, 0.1);
}

.dialog-notice strong {
  display: block;
  margin-bottom: 6px;
}

.dialog-notice p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.preview-panel {
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(31, 40, 48, 0.08);
}

.preview-head {
  display: grid;
  gap: 4px;
}

.preview-head span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.preview-head strong {
  font-size: 16px;
  line-height: 1.4;
}

.preview-media {
  min-height: 320px;
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(18, 23, 28, 0.96), rgba(31, 37, 42, 0.92));
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-media img {
  display: block;
  width: 100%;
  height: auto;
  object-fit: contain;
}

.dialog-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

@media (max-width: 820px) {
  .preview-grid {
    grid-template-columns: 1fr;
  }

  .preview-media {
    min-height: 220px;
  }
}
</style>
