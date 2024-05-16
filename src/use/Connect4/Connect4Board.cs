using System.Collections.Immutable;

namespace Connect4;

public sealed class Connect4Board
{
    readonly Stack<int> history = new();

    // bits     meaning
    // 0-2      height of column 0
    // 3-5      height of column 1
    // ...
    // 18-20    height of column 6
    // 21-26    column 0
    // ...
    // 57-62    column 6
    public ulong State { get; private set; }

    public int Player { get; private set; } = 1;
    public int Opponent { get; private set; } = 2;

    public int Height(int column)
    {
        if (column < 0 || column > 6) throw new IndexOutOfRangeException("The column index must be greater than or equal to zero and less than 7.");
        return (int)((State >> (3 * column)) & 7);
    }
    void SetHeight(int column, int height)
    {
        ulong h = (ulong)height << (3 * column);
        ulong mask = ~(ulong)(7 << (3 * column));
        State = (State & mask) | h;
        SetValidMoves();
    }
    public int this[int column, int row]
    {
        get
        {
            if (column < 0 || column > 6 || row < 0 || row >= Height(column)) return 0;
            return 1 + (int)((State >> (21 + 6 * column + row)) & 1);
        }
        private set
        {
            ulong b = (ulong)1 << (21 + 6 * column + row);
            if (value == 2)
                State |= b;
            else
                State &= ~b;
        }
    }

    public int Winner { get; private set; }
    public bool Full => ValidMoves.Length == 0;
    public bool Finished => Winner != 0 || Full;
    public bool CanUndo => history.Count > 0;

    public ImmutableArray<int> ValidMoves { get; private set; } = new int[] { 0, 1, 2, 3, 4, 5, 6 }.ToImmutableArray();
    void SetValidMoves() =>
        ValidMoves = Enumerable.Range(0, 7).Where(column => Height(column) < 6).ToImmutableArray();

    public void Move(int column)
    {
        if (Finished) throw new InvalidOperationException("The game already finished.");
        if (column < 0 || column > 6) throw new ArgumentOutOfRangeException(paramName: nameof(column), actualValue: column, message: "The move index must be greater or equal to zero and less than 7.");
        var height = Height(column);
        if (height > 5) throw new InvalidOperationException("This column is already full.");

        this[column, height] = Player;
        SetHeight(column, height+1);
        history.Push(column);

        try 
        {
            //
            // horizontal line
            //
            int count = 1;
            if (this[column-1, height] == Player)
            {
                count += 1;
                if (this[column-2, height] == Player)
                {
                    count +=1;
                    if (this[column-3, height] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }    
            }
            if (this[column+1, height] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column+2, height] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column+3, height] == Player)
                    {
                        Winner = Player; return;
                    }                
                }
            }

            //
            // vertical line
            //
            count = 1;
            if (this[column, height+1] == Player)
            {
                count += 1;
                if (this[column, height+2] == Player)
                {
                    count +=1;
                    if (this[column, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column, height - 3] == Player)
                    {
                        Winner = Player; return;
                    }
                }
            }

            //
            // upper left to lower right line
            //
            count = 1;
            if (this[column-1, height+1] == Player)
            {
                count += 1;
                if (this[column-2, height+2] == Player)
                {
                    count +=1;
                    if (this[column - 3, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column+1, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column+2, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column+3, height - 3] == Player)
                    {
                        Winner = Player; 
                        return;
                    }
                }
            }

            //
            // upper right to lower left line
            //
            count = 1;
            if (this[column+1, height+1] == Player)
            {
                count += 1;
                if (this[column + 2, height+2] == Player)
                {
                    count +=1;
                    if (this[column + 3, height + 3] == Player)
                    {
                        Winner = Player;
                        return;
                    }
                }
            }
            if (this[column-1, height - 1] == Player)
            {
                count +=1; if (count == 4) { Winner = Player; return; }
                if (this[column-2, height-2] == Player)
                {
                    count +=1; if (count == 4) { Winner = Player; return; }
                    if (this[column-3, height - 3] == Player)
                    {
                        Winner = Player; 
                        return;
                    }
                }
            }
        }
        finally
        {
            (Player, Opponent) = (Opponent, Player);
        }
    }
    public int Undo() 
    {
        if (history.Count == 0) return -1;
        var column = history.Pop();
        var height = Height(column);
        this[column, height-1] = 1;
        SetHeight(column, height-1);
        Winner = 0;
        (Player, Opponent) = (Opponent, Player);
        return column;
    }
    public Connect4Board Clone()
    {
        var copy = new Connect4Board
        {
            State = State,
            Player = Player,
            Opponent = Opponent,
            Winner = Winner
        };
        foreach (var column in history.Reverse()) copy.history.Push(column);
        return copy;
    }
}
