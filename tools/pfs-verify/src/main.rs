// pfs-verify: verification harness around b-inary/postflop-solver.
//
// Commands:
//   equity <flop> <oop_range> <ip_range>
//       Builds a checkdown-only flop tree (no bet sizes) and prints the
//       exact per-combo equity of each range on that flop:
//         EQUITY <player> <cards> <equity> <weight>
//       Equity is computed by the solver's own exhaustive chance
//       enumeration; nothing in the tree depends on strategies.
//
//   solve <flop> <oop_range> <ip_range> <pot> <stack> <bet> <raise> <iters>
//       Full postflop solve with the given bet abstraction. Prints:
//         EV <player> <chips>       (root EV, sums to the starting pot)
//         EXPLOITABILITY <chips>
//   flop: e.g. "Td9d6h"; ranges: their string syntax, weights allowed
//   ("AhAd:0.5,..."); bet/raise: e.g. "33%,60%,80%" and "2.5x".
use postflop_solver::*;
use std::env;
use std::io::{Read, Write};

fn build_game(flop: &str, oop: &str, ip: &str, pot: i32, stack: i32, bet: &str,
              raise: &str) -> Result<PostFlopGame, String> {
    let card_config = CardConfig {
        range: [oop.parse()?, ip.parse()?],
        flop: flop_from_str(flop)?,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    build_game_with_river(card_config, pot, stack, bet, raise)
}

fn build_game_river(board: &str, oop: &str, ip: &str, pot: i32, stack: i32,
                    bet: &str, raise: &str) -> Result<PostFlopGame, String> {
    // board: 6 or 10 card string (turn and river given).
    let b = board.trim();
    let cards: String = b.chars().filter(|c| !c.is_whitespace()).collect();
    let flop = flop_from_str(&cards[0..6])?;
    let turn = card_from_str(&cards[6..8])?;
    let river = card_from_str(&cards[8..10])?;
    let card_config = CardConfig {
        range: [oop.parse()?, ip.parse()?],
        flop,
        turn,
        river,
    };
    build_game_with_river(card_config, pot, stack, bet, raise)
}

fn build_game_with_river(card_config: CardConfig, pot: i32, stack: i32, bet: &str,
                         raise: &str) -> Result<PostFlopGame, String> {
    let bet_sizes = BetSizeOptions::try_from((bet, raise))?;
    let tree_config = TreeConfig {
        initial_state: if card_config.river != NOT_DEALT {
            BoardState::River
        } else if card_config.turn != NOT_DEALT {
            BoardState::Turn
        } else {
            BoardState::Flop
        },
        starting_pot: pot,
        effective_stack: stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };
    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("{e:?}"))?;
    PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{e:?}"))
}

fn cmd_equity(flop: &str, oop: &str, ip: &str) -> Result<(), String> {
    // No bet sizes anywhere: every line checks down, so per-hand EV is
    // exactly equity * pot (pot = 100). The solver's root expected_values
    // are computed through the game tree's chance nodes with proper card
    // handling, unlike the `equity()` helper, which linearizes around 0.5
    // and folds isomorphic suits (not comparable to exact equity).
    let mut game = build_game(flop, oop, ip, 100, 100, "", "")?;
    game.allocate_memory(false);
    solve(&mut game, 2, 0.0, false);
    game.cache_normalized_weights();
    for p in 0..2 {
        let hands = game.private_cards(p);
        let ev = game.expected_values(p);
        let w = game.normalized_weights(p);
        let names = holes_to_strings(hands)?;
        for i in 0..hands.len() {
            println!("EQUITY {} {} {:.9} {:.9}", p, names[i], ev[i] / 100.0, w[i]);
        }
    }
    Ok(())
}

fn cmd_solve(flop: &str, oop: &str, ip: &str, pot: i32, stack: i32, bet: &str,
             raise: &str, iters: u32, target: f32) -> Result<(), String> {
    let mut game = build_game(flop, oop, ip, pot, stack, bet, raise)?;
    run_solve(&mut game, pot, iters, target)
}

fn cmd_solve_board(board: &str, oop: &str, ip: &str, pot: i32, stack: i32, bet: &str,
                   raise: &str, iters: u32, target: f32) -> Result<(), String> {
    let mut game = build_game_river(board, oop, ip, pot, stack, bet, raise)?;
    run_solve(&mut game, pot, iters, target)
}

fn run_solve(game: &mut PostFlopGame, pot: i32, iters: u32, target: f32) -> Result<(), String> {
    game.allocate_memory(false);
    let expl = solve(game, iters, target, false);
    game.cache_normalized_weights();
    for p in 0..2 {
        let ev = game.expected_values(p);
        let w = game.normalized_weights(p);
        let mut wsum = 0.0;
        let mut evsum = 0.0;
        for i in 0..ev.len() {
            evsum += w[i] * ev[i];
            wsum += w[i];
        }
        println!("EV {} {:.6}", p, evsum / wsum);
    }
    println!("EXPLOITABILITY {:.6}", expl);
    Ok(())
}

// Binary streaming protocol, chunked (each chunk fits in pipe buffers to
// avoid deadlock): read u32 LE count, then count * 7 bytes of card indices
// (cuda-poker-solver encoding: card = suit*13 + rank), write count * 8
// bytes of little-endian u64 (their Hand::evaluate() strength). count == 0
// terminates.
fn cmd_eval7() -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();
    let mut count_buf = [0u8; 4];
    let mut cards_buf = [0u8; 7 * 4096];
    loop {
        reader.read_exact(&mut count_buf).map_err(|e| format!("read: {e}"))?;
        let count = u32::from_le_bytes(count_buf) as usize;
        if count == 0 {
            writer.flush().map_err(|e| format!("flush: {e}"))?;
            return Ok(());
        }
        reader.read_exact(&mut cards_buf[..count * 7]).map_err(|e| format!("read: {e}"))?;
        let mut out = vec![0u8; count * 8];
        for i in 0..count {
            let mut hand = Hand::new();
            for j in 0..7 {
                let c = cards_buf[i * 7 + j];
                let suit = (c / 13) as usize;
                let rank = (c % 13) as usize;
                // Their Card encoding: rank*4 + suit.
                hand = hand.add_card(rank * 4 + suit);
            }
            let strength = hand.evaluate() as u64;
            out[i * 8..i * 8 + 8].copy_from_slice(&strength.to_le_bytes());
        }
        writer.write_all(&out).map_err(|e| format!("write: {e}"))?;
        writer.flush().map_err(|e| format!("flush: {e}"))?;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let usage = "usage: pfs-verify equity <flop> <oop_range> <ip_range> | \
                 pfs-verify solve <flop> <oop_range> <ip_range> <pot> <stack> <bet> <raise> <iters> | \
                 pfs-verify solve-river <board10> <oop_range> <ip_range> <pot> <stack> <bet> <raise> <iters>";
    let res = match args.len() {
        2 if args[1] == "eval7" => cmd_eval7(),
        5 if args[1] == "equity" => {
            cmd_equity(&args[2], &args[3], &args[4])
        }
        10 if args[1] == "solve-river" => cmd_solve_board(
            &args[2], &args[3], &args[4],
            args[5].parse().unwrap(), args[6].parse().unwrap(),
            &args[7], &args[8], args[9].parse().unwrap(),
            0.0,
        ),
        11 if args[1] == "solve-river" => cmd_solve_board(
            &args[2], &args[3], &args[4],
            args[5].parse().unwrap(), args[6].parse().unwrap(),
            &args[7], &args[8], args[9].parse().unwrap(),
            args[10].parse().unwrap(),
        ),
        10 if args[1] == "solve" => cmd_solve(
            &args[2], &args[3], &args[4],
            args[5].parse().unwrap(), args[6].parse().unwrap(),
            &args[7], &args[8], args[9].parse().unwrap(),
            0.0,
        ),
        11 if args[1] == "solve" => cmd_solve(
            &args[2], &args[3], &args[4],
            args[5].parse().unwrap(), args[6].parse().unwrap(),
            &args[7], &args[8], args[9].parse().unwrap(),
            args[10].parse().unwrap(),
        ),
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
