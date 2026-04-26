<script setup lang="ts">
defineProps<{
  rows: Array<{
    imageId: number
    imageName: string
    thumbUrl: string | null
    volumeFractionLabel: string
    particleCountLabel: string
    meanSizeLabel: string
    unitLabel: string
    statusLabel: string
    statusType: 'success' | 'warning'
  }>
  currentPage: number
  pageSize: number
  total: number
  viewAllLabel?: string
}>()

const emit = defineEmits<{
  'update:currentPage': [value: number]
  'view-all': []
}>()
</script>

<template>
  <section class="glass-card image-check-card">
    <header class="image-check-card__head">
      <div>
        <h3 class="section-title">按图片检查</h3>
        <p class="section-subtitle">逐张核对分割结果对统计的贡献。</p>
      </div>
      <el-button plain :disabled="!total" @click="emit('view-all')">{{ viewAllLabel || '查看全部' }}</el-button>
    </header>

    <el-table :data="rows" stripe class="image-check-table" empty-text="当前模式暂无图片级统计结果">
      <el-table-column label="图片编号" min-width="220">
        <template #default="{ row }">
          <div class="thumbnail-cell">
            <div class="thumbnail-frame">
              <img v-if="row.thumbUrl" :src="row.thumbUrl" :alt="row.imageName" />
              <span v-else>SEM</span>
            </div>
            <strong>{{ row.imageName }}</strong>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="volumeFractionLabel" label="Vf" width="110" />
      <el-table-column prop="particleCountLabel" label="颗粒数" width="110" />
      <el-table-column prop="meanSizeLabel" label="平均尺寸（等效直径）" min-width="170" />
      <el-table-column prop="unitLabel" label="单位" width="90" />
      <el-table-column label="状态" width="140">
        <template #default="{ row }">
          <span class="status-tag" :class="row.statusType === 'success' ? 'is-success' : 'is-warning'">
            {{ row.statusLabel }}
          </span>
        </template>
      </el-table-column>
    </el-table>

    <div v-if="total > pageSize" class="image-check-card__footer">
      <el-pagination
        :current-page="currentPage"
        :page-size="pageSize"
        layout="prev, pager, next"
        :total="total"
        small
        @update:current-page="emit('update:currentPage', $event)"
      />
    </div>
  </section>
</template>

<style scoped>
.image-check-card {
  display: grid;
  gap: 16px;
  padding: 20px;
  border-radius: var(--stats-radius-lg, 20px);
  border: 1px solid var(--stats-border, #e5e7eb);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 248, 252, 0.95));
  box-shadow: var(--stats-shadow, 0 16px 30px rgba(15, 23, 42, 0.08));
}

.image-check-card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.thumbnail-cell {
  display: flex;
  align-items: center;
  gap: 12px;
}

.thumbnail-frame {
  width: 46px;
  height: 46px;
  overflow: hidden;
  display: grid;
  place-items: center;
  flex: 0 0 46px;
  border-radius: 12px;
  background: linear-gradient(135deg, #eef2f7, #dce4ef);
  border: 1px solid var(--stats-border, #e5e7eb);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
}

.thumbnail-frame img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: grayscale(1);
}

.thumbnail-frame span {
  color: #6b7280;
  font-size: 12px;
  font-weight: 700;
}

.status-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 88px;
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid transparent;
  letter-spacing: 0.01em;
}

.status-tag.is-success {
  background: #ecfdf3;
  border-color: #b7ebc8;
  color: #15803d;
}

.status-tag.is-warning {
  background: #fff7ed;
  border-color: #fed7aa;
  color: #c2410c;
}

.image-check-card__footer {
  display: flex;
  justify-content: center;
}

.image-check-table :deep(.el-table) {
  border-radius: 18px;
  overflow: hidden;
}

.image-check-table :deep(.el-table__header-wrapper th) {
  background: rgba(244, 247, 251, 0.94);
  color: var(--ink);
  font-weight: 700;
}

.image-check-table :deep(.el-table__row:hover > td) {
  background: rgba(47, 107, 255, 0.04) !important;
}

.image-check-table :deep(.el-button) {
  border-radius: 12px;
}
</style>
