interface ProgressState {
  completedLessons: Set<string>;
  toggleLesson: (lessonId: string) => void;
  isCompleted: (lessonId: string) => boolean;
  getModuleProgress: (lessonIds: string[]) => number;
}

// Simple zustand-like store using React state
// Using localStorage for persistence without a backend

const STORAGE_KEY = "ai-learning-progress";

const loadProgress = (): Set<string> => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return new Set(JSON.parse(saved));
  } catch {}
  return new Set();
};

const saveProgress = (completed: Set<string>) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...completed]));
};

import { useState, useCallback, useMemo, createContext, useContext, ReactNode } from "react";

const ProgressContext = createContext<ProgressState | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [completedLessons, setCompletedLessons] = useState<Set<string>>(loadProgress);

  const toggleLesson = useCallback((lessonId: string) => {
    setCompletedLessons((prev) => {
      const next = new Set(prev);
      if (next.has(lessonId)) next.delete(lessonId);
      else next.add(lessonId);
      saveProgress(next);
      return next;
    });
  }, []);

  const isCompleted = useCallback(
    (lessonId: string) => completedLessons.has(lessonId),
    [completedLessons]
  );

  const getModuleProgress = useCallback(
    (lessonIds: string[]) => {
      if (lessonIds.length === 0) return 0;
      const completed = lessonIds.filter((id) => completedLessons.has(id)).length;
      return Math.round((completed / lessonIds.length) * 100);
    },
    [completedLessons]
  );

  const value = useMemo(
    () => ({ completedLessons, toggleLesson, isCompleted, getModuleProgress }),
    [completedLessons, toggleLesson, isCompleted, getModuleProgress]
  );

  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>;
}

export function useProgress() {
  const ctx = useContext(ProgressContext);
  if (!ctx) throw new Error("useProgress must be used within ProgressProvider");
  return ctx;
}
