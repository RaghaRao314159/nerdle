"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

const MAX_GUESSES = 6;
const EQUATION_LENGTH = 8;
const ALLOWED_INPUT = new Set<string>(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "="]);

// Configuration: Set your custom solution here
// Set to null for random equations, or provide a specific equation
// const CUSTOM_SOLUTION: string | null = null;
// Examples:
const CUSTOM_SOLUTION = "3*4*6=72";
// const CUSTOM_SOLUTION = "10-6/2=7";
// const CUSTOM_SOLUTION = "12-3*4=0";

type TileState = "correct" | "present" | "absent";
type DisplayTileState = TileState | "pending" | "empty";
type GameStatus = "playing" | "won" | "lost";

const EQUATION_POOL = generateEquationPool();
const KEYBOARD_LAYOUT: string[][] = [
  ["7", "8", "9", "/"],
  ["4", "5", "6", "*"],
  ["1", "2", "3", "-"],
  ["Enter", "0", "+", "=", "Back"],
];

export default function Home() {
  const [solution, setSolution] = useState<string>(() => pickRandomEquation());
  const [guesses, setGuesses] = useState<string[]>([]);
  const [feedback, setFeedback] = useState<TileState[][]>([]);
  const [currentGuess, setCurrentGuess] = useState<string>("");
  const [gameStatus, setGameStatus] = useState<GameStatus>("playing");
  const [message, setMessage] = useState<string | null>(null);
  const [keyboardHints, setKeyboardHints] = useState<Record<string, TileState>>({});

  const submitGuess = useCallback(() => {
    if (gameStatus !== "playing") {
      return;
    }

    if (currentGuess.length !== EQUATION_LENGTH) {
      setMessage(`Equation must be exactly ${EQUATION_LENGTH} characters long.`);
      return;
    }

    const validation = validateGuess(currentGuess);
    if (!validation.valid) {
      setMessage(validation.message);
      return;
    }

    const evaluation = getTileStates(currentGuess, solution);
    const nextGuesses = [...guesses, currentGuess];
    const nextFeedback = [...feedback, evaluation];

    setGuesses(nextGuesses);
    setFeedback(nextFeedback);
    setKeyboardHints((prev) => mergeKeyboardHints(prev, currentGuess, evaluation));
    setCurrentGuess("");

    if (currentGuess === solution) {
      setGameStatus("won");
      setMessage("Exactly right! You solved the equation.");
      return;
    }

    if (nextGuesses.length >= MAX_GUESSES) {
      setGameStatus("lost");
      setMessage(`Out of guesses. The equation was ${solution}.`);
    }
  }, [currentGuess, feedback, gameStatus, guesses, solution]);

  const addCharacter = useCallback(
    (char: string) => {
      if (gameStatus !== "playing" || !ALLOWED_INPUT.has(char)) {
        return;
      }
      setCurrentGuess((prev) => {
        if (prev.length >= EQUATION_LENGTH) {
          return prev;
        }
        if (char === "=" && prev.includes("=")) {
          return prev;
        }
        return prev + char;
      });
    },
    [gameStatus],
  );

  const removeCharacter = useCallback(() => {
    if (gameStatus !== "playing") {
      return;
    }
    setCurrentGuess((prev) => prev.slice(0, -1));
  }, [gameStatus]);

  const resetGame = useCallback(() => {
    setSolution(pickRandomEquation());
    setGuesses([]);
    setFeedback([]);
    setCurrentGuess("");
    setGameStatus("playing");
    setMessage(null);
    setKeyboardHints({});
  }, []);

  useEffect(() => {
    if (!message || gameStatus !== "playing") {
      return;
    }
    const handle = window.setTimeout(() => setMessage(null), 2600);
    return () => window.clearTimeout(handle);
  }, [gameStatus, message]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key;

      if (key === "Enter") {
        event.preventDefault();
        if (gameStatus === "playing") {
          submitGuess();
        } else {
          resetGame();
        }
        return;
      }

      if (gameStatus !== "playing") {
        return;
      }

      if (key === "Backspace") {
        event.preventDefault();
        removeCharacter();
        return;
      }

      if (ALLOWED_INPUT.has(key)) {
        event.preventDefault();
        addCharacter(key);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [addCharacter, gameStatus, removeCharacter, resetGame, submitGuess]);

  const boardRows = useMemo(() => {
    return Array.from({ length: MAX_GUESSES }, (_, rowIndex) => {
      const guessForRow =
        guesses[rowIndex] ?? (rowIndex === guesses.length && gameStatus === "playing" ? currentGuess : "");
      const feedbackForRow = feedback[rowIndex] ?? [];

      return (
        <div key={`row-${rowIndex}`} className="grid w-full grid-cols-8 gap-2">
          {Array.from({ length: EQUATION_LENGTH }, (_, colIndex) => {
            const char = guessForRow[colIndex] ?? "";
            const state = feedbackForRow[colIndex]
              ? feedbackForRow[colIndex]
              : char
                ? "pending"
                : "empty";
            return (
              <span key={`tile-${rowIndex}-${colIndex}`} className={tileClassName(state)}>
                {char}
              </span>
            );
          })}
        </div>
      );
    });
  }, [currentGuess, feedback, gameStatus, guesses]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-3xl flex-col items-center px-4 py-10 sm:px-6">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Nerdle</h1>
          <p className="mt-2 text-sm text-slate-300 sm:text-base">
            Guess the hidden equation in {MAX_GUESSES} tries. Use digits together with + - * /.
          </p>
        </header>

        <section className="flex w-full flex-1 flex-col items-center gap-8">
          <div className="flex w-full flex-col items-center gap-2">{boardRows}</div>

          {message && (
            <div className="w-full rounded-md border border-slate-700 bg-slate-800/80 px-4 py-3 text-center text-sm sm:text-base">
              {message}
            </div>
          )}

          <div className="w-full space-y-2">
            {KEYBOARD_LAYOUT.map((row, rowIdx) => (
              <div key={`key-row-${rowIdx}`} className="flex w-full gap-2">
                {row.map((key) => {
                  const hint = keyboardHints[key];
                  const label = key === "Back" ? "Backspace" : key;
                  return (
                    <button
                      key={key}
                      type="button"
                      className={keyClassName(key, hint)}
                      onClick={() => {
                        if (key === "Enter") {
                          if (gameStatus === "playing") {
                            submitGuess();
                          } else {
                            resetGame();
                          }
                        } else if (key === "Back") {
                          removeCharacter();
                        } else {
                          addCharacter(key);
                        }
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          <footer className="w-full rounded-md border border-slate-800 bg-slate-900/80 px-4 py-3 text-xs text-slate-300 sm:text-sm">
            <p>
              Each guess must be a valid equation that evaluates correctly. Tiles turn green when the symbol is in the
              correct spot, purple when it is elsewhere in the solution, and black when it is not used.
            </p>
            {gameStatus !== "playing" && (
              <div className="mt-3 flex flex-col items-center gap-3 text-center">
                <span className="text-sm font-semibold text-slate-100 sm:text-base">
                  {gameStatus === "won"
                    ? "Nicely done! Ready for another round?"
                    : `Close one! The equation was ${solution}.`}
                </span>
                <button
                  type="button"
                  className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-400"
                  onClick={resetGame}
                >
                  Play again
                </button>
              </div>
            )}
          </footer>
        </section>
      </div>
    </div>
  );
}

function pickRandomEquation(): string {
  // If custom solution is set, use it instead of random
  if (CUSTOM_SOLUTION !== null) {
    return CUSTOM_SOLUTION;
  }
  
  // Otherwise, pick a random equation from the pool
  if (EQUATION_POOL.length === 0) {
    return "10+32=42";
  }
  const index = Math.floor(Math.random() * EQUATION_POOL.length);
  return EQUATION_POOL[index];
}

function generateEquationPool(): string[] {
  const results = new Set<string>();
  const operations: Array<{ symbol: string; calculate: (a: number, b: number) => number | null }> = [
    { symbol: "+", calculate: (a, b) => a + b },
    { symbol: "-", calculate: (a, b) => a - b },
    { symbol: "*", calculate: (a, b) => a * b },
    {
      symbol: "/",
      calculate: (a, b) => {
        if (b === 0 || a % b !== 0) {
          return null;
        }
        return a / b;
      },
    },
  ];

  const noLeadingZero = (value: number) => {
    const text = String(value);
    return text.length === 1 || !text.startsWith("0");
  };

  for (let left = 0; left < 100; left += 1) {
    if (!noLeadingZero(left)) {
      continue;
    }
    for (let right = 0; right < 100; right += 1) {
      if (!noLeadingZero(right)) {
        continue;
      }

      for (const op of operations) {
        const outcome = op.calculate(left, right);
        if (outcome === null || !Number.isInteger(outcome)) {
          continue;
        }
        if (!noLeadingZero(outcome)) {
          continue;
        }
        const equation = `${left}${op.symbol}${right}=${outcome}`;
        if (equation.length === EQUATION_LENGTH) {
          results.add(equation);
        }
      }
    }
  }

  return Array.from(results);
}

function validateGuess(guess: string): { valid: true } | { valid: false; message: string } {
  if (guess.length !== EQUATION_LENGTH) {
    return { valid: false, message: `Guesses must be ${EQUATION_LENGTH} characters.` };
  }

  if (![...guess].every((char) => ALLOWED_INPUT.has(char))) {
    return { valid: false, message: "Use only digits and the symbols + - * / =." };
  }

  if (guess.split("=").length !== 2) {
    return { valid: false, message: "Equation must contain exactly one equals sign." };
  }

  const [left, right] = guess.split("=") as [string, string];
  if (!left || !right) {
    return { valid: false, message: "Equation has to include values on both sides of =." };
  }

  if (!isExpressionValid(left) || !isExpressionValid(right)) {
    return { valid: false, message: "Use a well-formed equation (no consecutive operators)." };
  }

  const leftValue = safeEvaluate(left);
  const rightValue = safeEvaluate(right);

  if (leftValue === null || rightValue === null) {
    return { valid: false, message: "Equation could not be evaluated." };
  }

  if (!Number.isInteger(leftValue) || !Number.isInteger(rightValue)) {
    return { valid: false, message: "Stick to whole-number results." };
  }

  if (leftValue !== rightValue) {
    return { valid: false, message: "Equation does not balance. Try again!" };
  }

  return { valid: true };
}

function isExpressionValid(expression: string): boolean {
  if (!/^[0-9+\-*/]+$/.test(expression)) {
    return false;
  }
  if (!/\d/.test(expression[0]) || !/\d/.test(expression[expression.length - 1])) {
    return false;
  }

  let previousWasOperator = false;
  for (const char of expression) {
    const isOperator = char === "+" || char === "-" || char === "*" || char === "/";
    if (isOperator) {
      if (previousWasOperator) {
        return false;
      }
      previousWasOperator = true;
    } else {
      previousWasOperator = false;
    }
  }

  return true;
}

function safeEvaluate(expression: string): number | null {
  try {
    // eslint-disable-next-line no-new-func
    const fn = new Function(`return (${expression});`);
    const result = fn();
    if (typeof result !== "number" || Number.isNaN(result) || !Number.isFinite(result)) {
      return null;
    }
    return result;
  } catch {
    return null;
  }
}

function getTileStates(guess: string, target: string): TileState[] {
  const result: TileState[] = Array(EQUATION_LENGTH).fill("absent");
  const targetChars = target.split("");
  const used = Array(EQUATION_LENGTH).fill(false);

  for (let i = 0; i < EQUATION_LENGTH; i += 1) {
    if (guess[i] === target[i]) {
      result[i] = "correct";
      used[i] = true;
    }
  }

  for (let i = 0; i < EQUATION_LENGTH; i += 1) {
    if (result[i] === "correct") {
      continue;
    }
    const guessChar = guess[i];
    const matchIndex = targetChars.findIndex((char, idx) => !used[idx] && char === guessChar);
    if (matchIndex !== -1) {
      result[i] = "present";
      used[matchIndex] = true;
    }
  }

  return result;
}

function mergeKeyboardHints(
  prev: Record<string, TileState>,
  guess: string,
  evaluation: TileState[],
): Record<string, TileState> {
  const priority: Record<TileState, number> = { absent: 0, present: 1, correct: 2 };
  const next = { ...prev };

  guess.split("").forEach((char, index) => {
    const state = evaluation[index];
    const previousState = next[char];
    if (!previousState || priority[state] > priority[previousState]) {
      next[char] = state;
    }
  });

  return next;
}

function tileClassName(state: DisplayTileState): string {
  const base =
    "flex h-14 w-10 items-center justify-center rounded-md border-2 text-xl font-bold uppercase sm:h-16 sm:w-14 sm:text-2xl";
  switch (state) {
    case "correct":
      return `${base} border-green-600 bg-green-600 text-white`;
    case "present":
      return `${base} border-purple-600 bg-purple-600 text-white`;
    case "absent":
      return `${base} border-black bg-black text-slate-400`;
    case "pending":
      return `${base} border-sky-400 bg-slate-900 text-slate-100`;
    default:
      return `${base} border-slate-700 bg-slate-900 text-slate-500`;
  }
}

function keyClassName(key: string, state?: TileState): string {
  const base =
    "flex-1 rounded-md px-3 py-3 text-base font-semibold uppercase transition-colors duration-150 sm:text-lg";

  if (key === "Enter" || key === "Back") {
    return `${base} basis-[96px] bg-slate-700 text-slate-100 hover:bg-slate-600`;
  }

  if (!state) {
    return `${base} bg-slate-800 text-slate-100 hover:bg-slate-700`;
  }

  switch (state) {
    case "correct":
      return `${base} bg-green-600 text-white hover:bg-green-500`;
    case "present":
      return `${base} bg-purple-600 text-white hover:bg-purple-500`;
    default:
      return `${base} bg-black text-slate-300 hover:bg-slate-900`;
  }
}
